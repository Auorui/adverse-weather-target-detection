import torch
import torch.nn as nn
import torch.nn.functional as F
from D_YOLO.models.cbam import CBAM
from D_YOLO.models.odconv import ODConv2d

class FeatureAdaptionModule(nn.Module):
    """
    特征适配模块 (Feature Adaption Module)
    用于从清晰特征中学习有利信息，弥合Fc和Fd之间的差距
    """
    def __init__(self, in_channels, reduction=0.0625, kernel_num=4):
        """
        Args:
            in_channels: 输入通道数 Cin
            reduction: ODConv中的reduction比例
            kernel_num: ODConv中的卷积核数量
        """
        super(FeatureAdaptionModule, self).__init__()
        self.in_channels = in_channels
        self.odconv1 = ODConv2d(
            in_planes=in_channels,
            out_planes=in_channels,  # 1×Cin
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            reduction=reduction,
            kernel_num=kernel_num
        )
        self.odconv2 = ODConv2d(
            in_planes=in_channels,
            out_planes=2 * in_channels,  # 2×Cin
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            reduction=reduction,
            kernel_num=kernel_num
        )
        # CBAM, 卷积块注意力模块, 输出维度 1×Cin
        self.cbam = CBAM(in_planes=2 * in_channels, ratio=16, kernel_size=7)
        # 最后的1×1卷积将通道数从2×Cin恢复到1×Cin
        self.final_conv = nn.Conv2d(2 * in_channels, in_channels, 1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 [B, Cin, H, W]
        Returns:
            output: 适配后的特征图 [B, Cin, H, W]
        """
        # ODConv_1
        x = self.odconv1(x)  # [B, Cin, H, W]
        x = self.leaky_relu(x)
        # ODConv_2
        x = self.odconv2(x)  # [B, 2×Cin, H, W]
        x = self.leaky_relu(x)
        # CBAM注意力
        x = self.cbam(x)  # [B, 2×Cin, H, W]
        # 最终卷积恢复通道数
        x = self.final_conv(x)  # [B, Cin, H, W]
        x = self.leaky_relu(x)
        return x


class MultiScaleFeatureAdaption(nn.Module):
    """
    多尺度特征适配模块
    对三个尺度的特征分别应用FA模块，并使用加权KL散度损失
    """
    def __init__(self, channels_list, weights=[0.7, 0.2, 0.1], reduction=0.0625, kernel_num=4):
        """
        Args:
            channels_list: 三个尺度特征的通道数列表 [ch1, ch2, ch3]
            weights: 三个尺度的权重 [λ1, λ2, λ3]
            reduction: ODConv中的reduction比例
            kernel_num: ODConv中的卷积核数量
        """
        super(MultiScaleFeatureAdaption, self).__init__()

        self.weights = weights
        self.tau = 1.0  # 蒸馏温度

        # 为每个尺度创建独立的FA模块
        self.fa_modules = nn.ModuleList([
            FeatureAdaptionModule(channels, reduction, kernel_num)
            for channels in channels_list
        ])

    def forward(self, features):
        """
        前向传播

        Args:
            features: 输入特征图列表 [feat1, feat2, feat3]

        Returns:
            adapted_features: 适配后的特征图列表 [Fd1, Fd2, Fd3]
        """
        adapted_features = []
        for i, (fa_module, feat) in enumerate(zip(self.fa_modules, features)):
            adapted_feat = fa_module(feat)
            adapted_features.append(adapted_feat)

        return adapted_features

    def compute_kl_loss(self, adapted_features, clear_features):
        """
        计算多尺度KL散度损失
        Args:
            adapted_features: FA模块输出的特征图列表 [Fd1, Fd2, Fd3]
            clear_features: CFE模块输出的清晰特征图列表 [Fc1, Fc2, Fc3]

        Returns:
            total_loss: 加权后的总KL散度损失
        """
        total_loss = 0.0
        batch_size = adapted_features[0].size(0)

        for i, (Fd, Fc) in enumerate(zip(adapted_features, clear_features)):
            weight = self.weights[i] # 获取当前尺度的权重
            # 计算当前尺度的KL散度损失
            scale_loss = self._compute_channel_kl_divergence(Fc, Fd, self.tau)
            total_loss += weight * scale_loss # 加权累加
        return total_loss

    def _compute_channel_kl_divergence(self, Fc, Fd, tau=1.0):
        """
        计算通道级别的KL散度损失
        Args:
            Fc: 清晰特征图 [B, C, H, W]
            Fd: 适配特征图 [B, C, H, W]
            tau: 蒸馏温度
        Returns:
            kl_loss: KL散度损失
        """
        N, C, H, W = Fc.size()
        # 重塑为 [B, C, H×W]
        Fc_flat = Fc.view(N, C, -1)  # [B, C, H×W]
        Fd_flat = Fd.view(N, C, -1)  # [B, C, H×W]
        # 直接计算log_softmax，数值更稳定
        log_p_clear = F.log_softmax(Fc_flat / tau, dim=2)  # [B, C, H×W]
        log_p_adapted = F.log_softmax(Fd_flat / tau, dim=2)  # [B, C, H×W]
        # 计算p_clear (目标分布)
        p_clear = F.softmax(Fc_flat / tau, dim=2)  # [B, C, H×W]
        # KL散度: sum(p_clear * (log_p_clear - log_p_adapted))
        kl_per_channel = (p_clear * (log_p_clear - log_p_adapted)).sum(dim=2)  # [B, C]
        # 对batch和通道求平均，然后乘以τ²
        kl_loss = kl_per_channel.mean() * (tau ** 2)
        return kl_loss

    def update_temperature(self, temperature):
        """更新蒸馏温度"""
        self.tau = temperature
        # 同时更新所有ODConv的温度
        for fa_module in self.fa_modules:
            if hasattr(fa_module.odconv1, 'update_temperature'):
                fa_module.odconv1.update_temperature(temperature)
            if hasattr(fa_module.odconv2, 'update_temperature'):
                fa_module.odconv2.update_temperature(temperature)


# 测试代码
if __name__ == "__main__":
    # 测试FA模块
    print("测试单尺度FA模块...")
    in_channels = 128
    fa_module = FeatureAdaptionModule(in_channels)

    # 测试输入
    batch_size = 2
    input_feat = torch.randn(batch_size, in_channels, 32, 32)

    # 前向传播
    output_feat = fa_module(input_feat)
    print(f"输入特征图形状: {input_feat.shape}")
    print(f"输出特征图形状: {output_feat.shape}")

    # 测试多尺度FA模块
    print("\n测试多尺度FA模块...")
    channels_list = [128, 256, 512]  # 三个尺度的通道数
    weights = [0.7, 0.2, 0.1]  # 对应λ1, λ2, λ3

    multi_fa = MultiScaleFeatureAdaption(channels_list, weights)

    # 模拟三个尺度的输入特征
    multi_inputs = [
        torch.randn(batch_size, 128, 32, 32),  # stage4特征
        torch.randn(batch_size, 256, 16, 16),  # stage5特征
        torch.randn(batch_size, 512, 8, 8)  # stage6特征
    ]

    # 前向传播
    adapted_outputs = multi_fa(multi_inputs)
    print(f"输入特征尺度: {[feat.shape for feat in multi_inputs]}")
    print(f"输出特征尺度: {[feat.shape for feat in adapted_outputs]}")

    # 测试KL散度损失计算
    print("\n测试KL散度损失计算...")
    clear_features = [
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]

    kl_loss = multi_fa.compute_kl_loss(adapted_outputs, clear_features)
    print(f"多尺度KL散度损失: {kl_loss.item():.6f}")

    # 测试温度更新
    print("\n测试温度更新...")
    multi_fa.update_temperature(0.5)
    print(f"当前蒸馏温度: {multi_fa.tau}")