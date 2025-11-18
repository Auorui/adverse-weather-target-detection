import torch
import torch.nn as nn
import torch.nn.functional as F


class HazeAwareAttentionFusion(nn.Module):
    """
    雾霾感知注意力特征融合模块 (Haze-aware Attention Feature Fusion)
    用于融合雾霾特征和去雾特征
    """
    def __init__(self, channels, kernel_size=3, pool_size=3):
        """
        Args:
            channels: 输入通道数
            kernel_size: 卷积核大小
            pool_size: 平均池化核大小
        """
        super(HazeAwareAttentionFusion, self).__init__()
        self.channels = channels
        self.pool_size = pool_size
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        F_haze, F_dehaze = x
        X = F_haze + F_dehaze
        # T = X + UP(fconv(Pool(X)))
        pooled = self.avg_pool(X)
        conv_out = self.conv1(pooled)  # 3×3卷积
        upsampled = F.interpolate(conv_out, size=X.shape[2:], mode='bilinear', align_corners=False)
        T = X + upsampled
        attention_map = torch.sigmoid(T)
        F_h = self.conv2(F_haze)
        F_d = self.conv3(F_dehaze)
        F_fused = attention_map * F_d + (1 - attention_map) * F_h
        return F_fused


class MultiScaleAttentionFusion(nn.Module):
    """
    多尺度注意力特征融合模块
    对多个尺度的特征分别应用AFF模块
    """

    def __init__(self, channels_list, kernel_size=3, pool_size=3):
        super(MultiScaleAttentionFusion, self).__init__()

        self.fusion_modules = nn.ModuleList([
            HazeAwareAttentionFusion(channels, kernel_size, pool_size)
            for channels in channels_list
        ])

    def forward(self, haze_features, dehaze_features):
        """
        前向传播
        Args:
            haze_features: 雾霾特征列表 [feat1, feat2, feat3]
            dehaze_features: 去雾特征列表 [feat1, feat2, feat3]
        Returns:
            fused_features: 融合后的特征列表
        """
        fused_features = []
        for fusion_module, F_h, F_d in zip(self.fusion_modules, haze_features, dehaze_features):
            fused_feat = fusion_module([F_h, F_d])
            fused_features.append(fused_feat)
        return fused_features


if __name__ == "__main__":
    # 测试单尺度融合
    channels = 128
    batch_size = 2
    height, width = 32, 32

    aff_module = HazeAwareAttentionFusion(channels)

    # 模拟输入特征
    F_haze = torch.randn(batch_size, channels, height, width)
    F_dehaze = torch.randn(batch_size, channels, height, width)

    # 前向传播
    F_fused = aff_module([F_haze, F_dehaze])

    print(f"雾霾特征形状: {F_haze.shape}")
    print(f"去雾特征形状: {F_dehaze.shape}")
    print(f"融合特征形状: {F_fused.shape}")

    # 测试多尺度融合
    channels_list = [128, 256, 512]
    multi_aff = MultiScaleAttentionFusion(channels_list)

    haze_features = [
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]

    dehaze_features = [
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]

    fused_features = multi_aff(haze_features, dehaze_features)
    print(f"输入雾霾特征尺度: {[feat.shape for feat in haze_features]}")
    print(f"输入去雾特征尺度: {[feat.shape for feat in dehaze_features]}")
    print(f"输出融合特征尺度: {[feat.shape for feat in fused_features]}")