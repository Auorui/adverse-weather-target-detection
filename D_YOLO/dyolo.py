import torch
import torch.nn as nn
from D_YOLO.models.cfe import CFE
from D_YOLO.models.fa import MultiScaleFeatureAdaption
from D_YOLO.models.af import MultiScaleAttentionFusion

class DYOLO(nn.Module):
    """
    D-YOLO: 双分支去雾检测网络
    训练时：清晰分支(CFE) + 检测分支
    推理时：仅检测分支
    """
    def __init__(self,
                 detection_backbone='yolov8n',
                 cfe_weight_path='darknet53.pth',
                 adapt_channels=[128, 256, 512],
                 fusion_channels=[128, 256, 512],
                 fa_weights=[0.7, 0.2, 0.1]):
        """
        Args:
            detection_backbone: 检测骨干网络类型
            cfe_weight_path: CFE预训练权重路径
            adapt_channels: 特征适配通道数 [stage4, stage5, stage6]
            fusion_channels: 融合模块通道数
            fa_weights: 特征适配权重 [λ1, λ2, λ3]
        """
        super(DYOLO, self).__init__()
        # 清晰特征提取网络 (仅训练时使用)
        self.cfe = CFE(weight_path=cfe_weight_path, adapt_channels=adapt_channels)
        # 特征适配模块
        self.feature_adaption = MultiScaleFeatureAdaption(
            channels_list=adapt_channels,
            weights=fa_weights
        )