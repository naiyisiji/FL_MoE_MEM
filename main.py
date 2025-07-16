import argparse
import torch
import numpy as np
import random
from train import train_two_stages

def main():
    parser = argparse.ArgumentParser(description='联邦学习训练')
    # 基本设置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='使用的设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dataset', type=str, default='cifar10', help='使用的数据集')
    parser.add_argument('--data_percentage', type=float, default=1.0, help='使用的数据百分比')

    # 模型设置
    parser.add_argument('--use_moe', default=True, help='是否使用混合专家模型')
    parser.add_argument('--encoder_type', type=str, default='resnet', help='编码器类型')
    parser.add_argument('--moe_gate_type', type=str, default='mlp', help='MOE门控类型')
    parser.add_argument('--local_expert_type', type=str, default='linear', help='本地专家类型')
    parser.add_argument('--model_name', type=str, default='best_model', help='保存的最佳模型名称')

    # 训练设置
    parser.add_argument('--num_clients', type=int, default=10, help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=100, help='通信轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='每个客户端的本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')

    # 聚合设置
    parser.add_argument('--encoder_agg_method', type=str, default='FedAvg', help='编码器聚合方法')
    parser.add_argument('--pred_head_agg_method', type=str, default='FedAvg', help='预测头聚合方法')
    parser.add_argument('--moe_agg_method', type=str, default='FedAvg', help='MOE门控聚合方法')

    # MOE设置
    parser.add_argument('--a_weight', type=float, default=0.5, help='MOE中loss1的权重')
    parser.add_argument('--b_weight', type=float, default=0.5, help='MOE中loss2的权重')
    parser.add_argument('--top_k', type=int, default=2, help='moe_gate开启的topk个专家')

    # 数据划分方法
    parser.add_argument('--data_split_method', type=str, default='dirichlet', choices=['iid', 'non-iid', 'dirichlet'], help='数据划分方法')

    # 学习率衰减设置
    parser.add_argument('--lr_decay_step', type=int, default=20, help='学习率衰减的步数')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help='学习率衰减的因子')

    # 聚类设置
    parser.add_argument('--percent', type=float, default=0.5, help='用于初始化阈值的百分比参数')
    # 补充第二阶段聚类与标签更新的超参数
    parser.add_argument('--plabel', type=float, default=0.5, help='伪标签比例参数')
    parser.add_argument('--r1', type=float, default=1.0, help='聚类更新阶段的比率参数1')
    parser.add_argument('--r2', type=float, default=1.0, help='聚类更新阶段的比率参数2')
    parser.add_argument('--r3', type=float, default=1.0, help='聚类更新阶段的比率参数3')
    parser.add_argument('--t1', type=float, default=0.5, help='聚类更新阶段的阈值参数1')
    parser.add_argument('--t2', type=float, default=0.5, help='聚类更新阶段的阈值参数2')

    args = parser.parse_args()

    # 设置随机种子以确保结果可复现
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 开始训练
    server = train_two_stages(args)

if __name__ == "__main__":
    main()
