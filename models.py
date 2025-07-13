import torch
import torch.nn as nn
import torchvision.models as models

def create_encoder(encoder_type, pretrained=False, output_dim=512):
    """创建编码器模型"""
    if encoder_type.lower() == 'resnet':
        # 使用完整的ResNet18，移除最后的全连接层
        resnet = models.resnet18(pretrained=pretrained)
        return nn.Sequential(*list(resnet.children())[:-1])  # 移除最后一层(fc)
    elif encoder_type.lower() == 'mlp':
        # 使用多层感知机
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    else:
        raise ValueError(f"不支持的编码器类型: {encoder_type}")

def create_moe_gate(moe_gate_type, input_dim, num_experts=5):
    """创建MOE门控模型"""
    if moe_gate_type.lower() == 'mlp':
        # 使用多层感知机作为门控
        return nn.Sequential(
            nn.Flatten(),  # 确保输入被展平为2D张量
            nn.Linear(input_dim, 64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )
    else:
        raise ValueError(f"不支持的MOE门控类型: {moe_gate_type}")

def create_local_expert(local_expert_type, input_dim, output_dim=512):
    """创建本地专家模型"""
    if local_expert_type.lower() == 'linear':
        # 使用线性层作为本地专家，保持4D张量格式
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1),  # 1x1卷积替代线性层
            nn.ReLU()
        )
    elif local_expert_type.lower() == 'mlp':
        # 使用多层感知机作为本地专家，保持4D张量格式
        return nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1),  # 1x1卷积替代线性层
            nn.ReLU(),
            nn.Conv2d(128, output_dim, kernel_size=1),  # 1x1卷积替代线性层
            nn.ReLU()
        )
    else:
        raise ValueError(f"不支持的本地专家类型: {local_expert_type}")

def create_pred_head(pred_head_type, input_dim, output_dim=10):
    """创建预测头模型"""
    if pred_head_type.lower() == 'resnet':
        # 对应ResNet编码器的预测头，处理4D特征图
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(input_dim, output_dim)
        )
    else:
        # 通用预测头，处理2D特征向量
        return nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
