import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, models, transforms
import numpy as np
import random
from typing import List

def setup(rank, world_size, port=12355):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置GPU设备
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_model(model_name: str = "resnet18", num_classes: int = 10, pretrained: bool = True):
    if model_name.lower() == "resnet18":
        # 根据pretrained参数决定是否加载预训练权重
        model = models.resnet18(pretrained=pretrained)
        backbone = nn.Sequential(*list(model.children())[:-1])
        decoder = nn.Linear(model.fc.in_features, num_classes)
        gate = nn.Linear(model.fc.in_features, 1)  # 可学门控模型
        return backbone, decoder, gate
    elif model_name.lower() == "simple_cnn":
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super(SimpleCNN, self).__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(32 * 7 * 7, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes)
                )
                self.gate = nn.Linear(32 * 7 * 7, 1)  # 可学门控模型

            def forward(self, x):
                x = self.backbone(x)
                x = x.view(-1, 32 * 7 * 7)
                gate_output = torch.sigmoid(self.gate(x))
                output = self.decoder(x)
                return output, gate_output

        backbone = SimpleCNN().backbone
        decoder = SimpleCNN().decoder
        gate = SimpleCNN().gate
        return backbone, decoder, gate
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
class DataSplitter:
    @staticmethod
    def iid_split(dataset: Dataset, num_clients: int) -> List[List[int]]:
        num_samples = len(dataset) // num_clients
        client_indices = [[] for _ in range(num_clients)]
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i in range(num_clients):
            client_indices[i] = indices[i * num_samples : (i + 1) * num_samples]
        
        return client_indices
    
    @staticmethod
    def non_iid_split(dataset: Dataset, num_clients: int, num_classes_per_client: int = 2) -> List[List[int]]:
        client_indices = [[] for _ in range(num_clients)]
        
        # 假设数据集有一个targets属性存储标签
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
        else:
            # 如果没有targets属性，遍历数据集获取标签
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
        classes = np.unique(targets)
        num_classes = len(classes)
        
        # 为每个客户端分配特定的类别
        class_assignments = []
        for i in range(num_clients):
            client_classes = random.sample(list(classes), num_classes_per_client)
            class_assignments.append(client_classes)
        
        # 为每个类别创建索引列表
        class_indices = {c: np.where(targets == c)[0].tolist() for c in classes}
        
        # 为每个客户端分配样本
        for i in range(num_clients):
            client_classes = class_assignments[i]
            for c in client_classes:
                # 每个类别分配相等数量的样本
                num_samples_per_class = len(class_indices[c]) // num_clients
                client_indices[i].extend(class_indices[c][i * num_samples_per_class : (i + 1) * num_samples_per_class])
        
        return client_indices
    
    @staticmethod
    def dirichlet_split(dataset: Dataset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
        else:
            targets = np.array([dataset[i][1] for i in range(len(dataset))])
        
        classes = np.unique(targets)
        num_classes = len(classes)
        
        client_dist = np.random.dirichlet([alpha] * num_clients, num_classes)
        
        client_indices = [[] for _ in range(num_clients)]
        
        for c in classes:
            class_indices = np.where(targets == c)[0]
            np.random.shuffle(class_indices)
            
            proportions = client_dist[c]
            cumulative = np.cumsum(proportions)
            cumulative = np.round(cumulative * len(class_indices)).astype(int)
            cumulative = np.insert(cumulative, 0, 0)
            
            for i in range(num_clients):
                client_indices[i].extend(class_indices[cumulative[i]:cumulative[i+1]])
        
        return client_indices

def run_federated_learning(rank, server_instance, client_instance, args):
    if args.num_gpus > 1:
        setup(rank, args.num_gpus, port=12355 + rank)
    # device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if args.dataset.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        num_classes = 10
        # 对于MNIST，如果使用resnet，需要调整输入通道数
        if args.model.lower() == "resnet18":
            backbone, decoder, gate = create_model(args.model, num_classes)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif args.dataset.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 分割数据集
    if args.split_method.lower() == "iid":
        client_indices = DataSplitter.iid_split(train_dataset, args.num_clients)
    elif args.split_method.lower() == "non-iid":
        client_indices = DataSplitter.non_iid_split(train_dataset, args.num_clients)
    elif args.split_method.lower() == "dirichlet":
        client_indices = DataSplitter.dirichlet_split(train_dataset, args.num_clients, alpha=args.alpha)
    else:
        raise ValueError(f"不支持的数据分割方法: {args.split_method}")
    
    # 创建全局测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # 创建模型
    if args.dataset.lower() == "mnist" and args.model.lower() == "resnet18":
        # 已经在上面调整了MNIST的resnet输入通道
        global_backbone = backbone
        global_decoder = decoder
        global_gate = gate
    else:
        global_backbone, global_decoder, global_gate = create_model(args.model, num_classes)
    
    # 公平分配GPU资源
    total_gpus = min(args.num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    
    # 计算服务器和客户端应分配的GPU比例
    server_gpu_share = min(args.server_gpus, total_gpus)
    client_gpu_share = total_gpus - server_gpu_share
    
    if client_gpu_share < 1:
        # 如果没有足够的GPU给客户端，调整分配
        server_gpu_share = 1
        client_gpu_share = max(1, total_gpus - 1)
    
    # 分配GPU给服务器和客户端
    gpu_indices = list(range(total_gpus))
    
    # 循环分配GPU，确保公平性
    server_gpus = []
    client_gpu_assignments = [[] for _ in range(args.num_clients)]
    
    # 先分配服务器的GPU
    for i in range(server_gpu_share):
        server_gpus.append(gpu_indices[i % total_gpus])
    
    # 再分配客户端的GPU
    client_idx = 0
    for i in range(server_gpu_share, total_gpus):
        client_gpu_assignments[client_idx].append(gpu_indices[i])
        client_idx = (client_idx + 1) % args.num_clients
    
    # 如果客户端数量多于剩余GPU，循环分配
    if args.num_clients > client_gpu_share and client_gpu_share > 0:
        for i in range(args.num_clients):
            if not client_gpu_assignments[i]:
                client_gpu_assignments[i].append(gpu_indices[i % client_gpu_share])
    
    # 服务器只在第一个进程上运行
    if rank == 0:
        print(f"服务器使用GPU: {server_gpus}")
        server = server_instance(global_backbone, global_decoder, server_gpus, args.aggregation, use_moe=args.use_moe, moe_strategy=args.moe_strategy)
        
        # 记录全局模型性能
        server.evaluate(test_loader)
    
    # 客户端在所有进程上分布
    clients_per_process = args.num_clients // args.num_gpus
    start_client = rank * clients_per_process
    end_client = (rank + 1) * clients_per_process if rank < args.num_gpus - 1 else args.num_clients
    
    clients = []
    for i in range(start_client, end_client):
        # 为每个客户端创建本地数据集
        client_train_data = Subset(train_dataset, client_indices[i])
        client_test_data = test_dataset  # 所有客户端使用相同的测试集
        
        # 每个客户端有自己的模型副本
        client_backbone, client_decoder, client_gate = create_model(args.model, num_classes)
        if args.dataset.lower() == "mnist" and args.model.lower() == "resnet18":
            client_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 创建客户端，分配对应的GPU
        print(f"客户端 {i} 使用GPU: {client_gpu_assignments[i]}")
        client = client_instance(
            client_id=i,
            train_data=client_train_data,
            test_data=client_test_data,
            backbone=client_backbone,
            decoder=client_decoder,
            gate=client_gate,
            device_ids=client_gpu_assignments[i],
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.local_epochs,
            optimizer=args.optimizer,
            criterion=nn.CrossEntropyLoss(),
            use_moe=args.use_moe,
            moe_strategy=args.moe_strategy
        )
        clients.append(client)
    
    # 联邦学习训练循环
    for round in range(args.global_rounds):
        print(f"\n===== 全局轮次 {round+1}/{args.global_rounds} =====")
        
        # 服务器广播全局模型
        if rank == 0:
            broadcast_params = server.broadcast()
            global_backbone_params = broadcast_params['global_backbone']
            if args.use_moe:
                expert_decoder_params = broadcast_params['expert_decoder']
        
        # 同步全局模型到所有GPU
        if args.num_gpus > 1:
            for name, param in global_backbone_params.items():
                dist.broadcast(param, src=0)
            if args.use_moe:
                if args.moe_strategy == 'single_expert':
                    for name, param in expert_decoder_params.items():
                        dist.broadcast(param, src=0)
                elif args.moe_strategy == 'multi_expert':
                    for expert in expert_decoder_params:
                        for name, param in expert.items():
                            dist.broadcast(param, src=0)
        
        # 客户端训练
        client_backbone_updates = []
        client_decoder_updates = []
        client_sizes = []
        
        for client in clients:
            # 更新客户端模型为全局模型
            client_backbone_state = client.backbone.state_dict()
            for name, param in global_backbone_params.items():
                if name in client_backbone_state:
                    client_backbone_state[name] = param
            client.backbone.load_state_dict(client_backbone_state)

            if args.use_moe and round > 0:  # 第一次全局训练后才更新专家模型
                client.update_expert_model(expert_decoder_params)
            
            # 本地训练
            try:
                client_backbone_params, client_decoder_params = client.train()
                client_backbone_updates.append(client_backbone_params)
                client_decoder_updates.append(client_decoder_params)
                client_sizes.append(len(client.train_data))
            except Exception as e:
                print(f"客户端 {client.client_id} 训练出错: {e}")
                # 可以选择跳过此客户端或采取其他恢复措施
        
        # 收集所有GPU上的客户端更新
        all_backbone_updates = [None] * args.num_gpus
        all_decoder_updates = [None] * args.num_gpus
        all_sizes = [None] * args.num_gpus
        
        if args.num_gpus > 1:
            # 使用all_gather收集所有进程的客户端更新
            dist.all_gather_object(all_backbone_updates, client_backbone_updates)
            dist.all_gather_object(all_decoder_updates, client_decoder_updates)
            dist.all_gather_object(all_sizes, client_sizes)
        else:
            all_backbone_updates[0] = client_backbone_updates
            all_decoder_updates[0] = client_decoder_updates
            all_sizes[0] = client_sizes
        
        # 展平收集到的更新
        flattened_backbone_updates = [update for process_updates in all_backbone_updates if process_updates is not None for update in process_updates]
        flattened_decoder_updates = [update for process_updates in all_decoder_updates if process_updates is not None for update in process_updates]
        flattened_sizes = [size for process_sizes in all_sizes if process_sizes is not None for size in process_sizes]
        
        # 服务器聚合模型
        if rank == 0:
            if not flattened_backbone_updates:
                print("警告: 没有收集到任何客户端更新，跳过本轮聚合")
                continue
                
            server.aggregate(flattened_backbone_updates, flattened_decoder_updates, flattened_sizes)
            # 评估全局模型
            server.evaluate(test_loader)
        
        # 标记第一次全局训练已完成
        if args.use_moe and round == 0:
            for client in clients:
                client.first_global_round_completed = True
    
    if args.num_gpus > 1:
        cleanup()
