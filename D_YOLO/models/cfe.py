import torch
import torch.nn as nn
from D_YOLO.models.darknet53 import DarkNet53
from D_YOLO.models.cspdarknet53 import CspDarkNet53

def load_weights(model, weight_path='darknet53.pth'):
    """
    专门为DarkNet53设计的权重加载函数
    可以处理包含FC层的预训练权重
    """
    # 加载预训练权重
    checkpoint = torch.load(weight_path, map_location='cpu')
    # 处理不同的权重文件格式
    if 'state_dict' in checkpoint:
        pretrained_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_dict = checkpoint['model']
    else:
        pretrained_dict = checkpoint
    # 获取当前模型的状态字典
    model_dict = model.state_dict()
    # 只保留骨干网络部分的权重
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        # 移除DataParallel的'module.'前缀
        if k.startswith('module.'):
            k = k[7:]
        # 只保留我们需要的层（跳过FC层等）
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
            print(f"✓ 加载层: {k}")
        else:
            # 特别处理一些可能的名称变化
            if 'fc' in k or 'classifier' in k or 'avgpool' in k:
                print(f"✗ 跳过分类层: {k}")
            else:
                print(f"✗ 跳过不匹配层: {k}")
    # 更新模型权重
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"\n权重加载完成: {len(filtered_dict)}/{len(pretrained_dict)} 层成功加载")
    return model

class CFE(nn.Module):
    """
    清晰特征提取网络 (Clear Feature Extraction Network)
    仅在训练阶段激活，用于从清晰图像中提取特征
    """
    def __init__(self, weight_path='darknet53.pth', adapt_channels=[256, 512, 1024]):
        """
        Args:
            weight_path: 预训练权重路径
            adapt_channels: 三个尺度特征图的目标通道数 [stage4, stage5, stage6]
        """
        super(CFE, self).__init__()
        # DarkNet53骨干网络
        if weight_path.split('.')[-2] == 'darknet53':
            self.backbone = DarkNet53(num_classes=1000)
        elif weight_path.split('.')[-2] == 'cspdarknet53':
            self.backbone = CspDarkNet53(num_classes=1000)
        if weight_path:
            self.backbone = load_weights(self.backbone, weight_path)
        # 1x1卷积层用于调整通道数，使其与FA模块兼容
        # 输入通道数: [256, 512, 1024] 对应 stage4, stage5, stage6
        self.conv1x1_stage4 = nn.Conv2d(256, adapt_channels[0], 1, stride=1, padding=0)
        self.conv1x1_stage5 = nn.Conv2d(512, adapt_channels[1], 1, stride=1, padding=0)
        self.conv1x1_stage6 = nn.Conv2d(1024, adapt_channels[2], 1, stride=1, padding=0)
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        # 初始化1x1卷积权重
        self._init_conv1x1_weights()

    def _init_conv1x1_weights(self):
        """初始化1x1卷积层的权重"""
        for m in [self.conv1x1_stage4, self.conv1x1_stage5, self.conv1x1_stage6]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入清晰图像 [batch_size, 3, H, W]
        Returns:
            features: 三个尺度的清晰特征图 [Fc4, Fc5, Fc6]
        """
        # 通过DarkNet53骨干网络提取多尺度特征
        stage4, stage5, stage6 = self.backbone(x)
        # 通过1x1卷积调整通道数
        Fc4 = self.leaky_relu(self.conv1x1_stage4(stage4))  # [B, adapt_channels[0], H/8, W/8]
        Fc5 = self.leaky_relu(self.conv1x1_stage5(stage5))  # [B, adapt_channels[1], H/16, W/16]
        Fc6 = self.leaky_relu(self.conv1x1_stage6(stage6))  # [B, adapt_channels[2], H/32, W/32]
        return [Fc4, Fc5, Fc6]

    def freeze_backbone(self):
        """冻结骨干网络权重，只训练1x1卷积层"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("CFE骨干网络已冻结")

    def unfreeze_backbone(self):
        """解冻骨干网络权重"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("CFE骨干网络已解冻")

    def get_trainable_parameters(self):
        """获取需要训练的参数"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

if __name__ == "__main__":
    cfe = CFE(weight_path='cspdarknet53.pth', adapt_channels=[128, 256, 512])
    cfe.eval()

    print("CFE模块结构:")
    print(f"Stage4 1x1卷积: {cfe.conv1x1_stage4}")
    print(f"Stage5 1x1卷积: {cfe.conv1x1_stage5}")
    print(f"Stage6 1x1卷积: {cfe.conv1x1_stage6}")

    # 测试前向传播
    with torch.no_grad():
        # 模拟输入清晰图像
        batch_size = 2
        input_size = 256
        clear_image = torch.randn(batch_size, 3, input_size, input_size)

        # 提取清晰特征
        features = cfe(clear_image)

        print(f"\n输入图像形状: {clear_image.shape}")
        print(f"输出特征图数量: {len(features)}")
        print(f"Fc4 特征图形状: {features[0].shape}")  # [2, 128, 32, 32]
        print(f"Fc5 特征图形状: {features[1].shape}")  # [2, 256, 16, 16]
        print(f"Fc6 特征图形状: {features[2].shape}")  # [2, 512, 8, 8]

        # 测试冻结功能
        cfe.freeze_backbone()
        trainable_params = cfe.get_trainable_parameters()
        print(f"可训练参数数量: {len(trainable_params)}")