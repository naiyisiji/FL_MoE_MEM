import torch
import torch.multiprocessing as mp
import argparse
from train import run_federated_learning
from server import Server
from client import Client

def main():
    parser = argparse.ArgumentParser(description='联邦学习框架')

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet18', 
                        help='使用的模型 (resnet18, simple_cnn)')

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        help='使用的数据集 (MNIST, CIFAR10)')
    parser.add_argument('--split-method', type=str, default='dirichlet', 
                        help='数据分割方法 (iid, non-iid, dirichlet)')
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Dirichlet分布的参数，控制数据非IID程度')

    # 客户端参数
    parser.add_argument('--num-clients', type=int, default=10, 
                        help='客户端数量')
    parser.add_argument('--local-epochs', type=int, default=5, 
                        help='每个客户端的本地训练轮次')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help='优化器 (sgd, adam)')

    # 服务器参数
    parser.add_argument('--aggregation', type=str, default='fedavg', 
                        help='聚合方法 (fedavg)')
    parser.add_argument('--global-rounds', type=int, default=10, 
                        help='全局训练轮次')
    parser.add_argument('--server-gpus', type=int, default=1, 
                        help='服务器希望使用的GPU数量，实际会根据总资源调整')

    # 分布式训练参数
    parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count(), 
                        help='使用的GPU总数量')

    # MoE参数
    parser.add_argument('--use-moe', action='store_true', default=True,
                        help='是否使用MoE训练')
    parser.add_argument('--moe-strategy', type=str, default='multi_expert', 
                        help='专家学习策略 (single_expert, multi_expert)')

    args = parser.parse_args()

    print(f"使用 {args.num_gpus} 个GPU进行分布式训练")
    print(f"客户端数量: {args.num_clients}")

    if args.num_gpus > 1 and torch.cuda.is_available():
        # 多GPU分布式训练
        mp.spawn(run_federated_learning,
                 args=(Server, Client, args),
                 nprocs=args.num_gpus,
                 join=True)
    else:
        # 单GPU或CPU训练
        run_federated_learning(0, Server, Client, args)

if __name__ == "__main__":
    main()
