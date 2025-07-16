import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from server import Server
from client import Client
from models import create_encoder, create_moe_gate, create_local_expert, create_pred_head

def moe_pretrain(args):
    # 创建工作目录和最佳模型文件夹
    work_dir = os.path.join(os.getcwd(), 'work_dir')
    best_model_dir = os.path.join(work_dir, 'best_model')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # 根据百分比选择数据
    if args.data_percentage < 1.0:
        num_train_samples = int(len(trainset) * args.data_percentage)
        num_test_samples = int(len(testset) * args.data_percentage)
        
        train_indices = np.random.choice(len(trainset), num_train_samples, replace=False)
        test_indices = np.random.choice(len(testset), num_test_samples, replace=False)
        
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
    
    # 使用相同的数据划分方法划分测试集
    test_datasets = split_data(testset, args.num_clients, args.data_split_method)
    testloaders = [DataLoader(tds, batch_size=args.batch_size, shuffle=False) 
                  for tds in test_datasets]
    
    # 创建全局测试数据加载器
    global_testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    encoder_output_dim = 512
    output_dim = 10
    
    # 创建Server
    server_encoder = create_encoder(args.encoder_type, pretrained=False).to(args.device)
    server_moe_gate = create_moe_gate(args.moe_gate_type, encoder_output_dim, args.num_clients).to(args.device)
    server_local_expert = create_local_expert(args.local_expert_type, encoder_output_dim, encoder_output_dim).to(args.device)
    server_pred_head = create_pred_head(args.encoder_type, encoder_output_dim, output_dim).to(args.device)
    
    server = Server(
        encoder=server_encoder,
        moe_gate=server_moe_gate,
        local_expert=server_local_expert,
        pred_head=server_pred_head,
        num_clients=args.num_clients,
        encoder_agg_method=args.encoder_agg_method,
        moe_agg_method=args.moe_agg_method,
        pred_head_agg_method=args.pred_head_agg_method,
        device=args.device,
        use_moe=args.use_moe
    )
    
    # 创建Clients
    clients = []
    client_datasets = split_data(trainset, args.num_clients, args.data_split_method)
    
    for client_id in range(args.num_clients):
        client_encoder = create_encoder(args.encoder_type, pretrained=False).to(args.device)
        client_moe_gate = create_moe_gate(args.moe_gate_type, encoder_output_dim, args.num_clients).to(args.device)
        client_local_expert = create_local_expert(args.local_expert_type, encoder_output_dim, encoder_output_dim).to(args.device)
        client_pred_head = create_pred_head(args.encoder_type, encoder_output_dim, output_dim).to(args.device)
        
        client = Client(
            client_id=client_id,
            encoder=client_encoder,
            moe_gate=client_moe_gate,
            local_expert=client_local_expert,
            pred_head=client_pred_head,
            train_data=client_datasets[client_id],
            batch_size=args.batch_size,
            device=args.device,
            use_moe=args.use_moe,
            a_weight=args.a_weight,
            b_weight=args.b_weight,
            top_k=args.top_k,
            lr=args.lr
        )
        clients.append(client)
    
    # 设置学习率调度器
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.StepLR(server.optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    server.set_lr_scheduler(scheduler)
    
    # 初始化最佳准确率和模型保存路径
    best_accuracy = 0.0
    model_params_path = os.path.join(best_model_dir, 'model_params.pth')
    info_json_path = os.path.join(best_model_dir, 'info.json')
    
    # 联邦学习训练循环
    for round in range(args.num_rounds):
        print(f"\n=== 第 {round+1}/{args.num_rounds} 轮 ===")
        
        # 客户端本地训练和评估
        client_updates = []
        client_test_results = []  # 收集客户端测试结果
        
        for client_id, client in enumerate(clients):
            # 从服务器获取全局模型参数
            global_encoder = server.get_encoder()
            global_moe_gate = server.get_moe_gate()
            global_local_expert = server.get_local_expert()
            global_pred_head = server.get_pred_head()
            experts, one_hot_vector = server.get_experts_and_vector(client.client_id)
            
            # 设置客户端模型参数
            client.set_encoder(global_encoder)
            client.set_moe_gate(global_moe_gate)
            client.set_local_expert(global_local_expert)
            client.set_pred_head(global_pred_head)
            client.set_experts(experts, one_hot_vector)
            
            # 本地训练
            print(f"客户端 {client.client_id} 本地训练中...")
            encoder, moe_gate, local_expert, pred_head = client.train(args.local_epochs)
            
            # 评估本地模型并保存结果
            local_loss, local_acc = client.evaluate(testloaders[client_id])
            print(f"客户端 {client.client_id} 本地模型测试结果: Loss = {local_loss:.4f}, Acc = {local_acc:.4f}")
            client_test_results.append({
                "client_id": client.client_id,
                "test_loss": float(local_loss),
                "test_accuracy": float(local_acc)
            })
            
            # 保存客户端更新
            client_updates.append({
                'client_id': client.client_id,
                'encoder': encoder,
                'moe_gate': moe_gate,
                'local_expert': local_expert,
                'pred_head': pred_head
            })
        
        # 服务器聚合模型参数
        server.aggregate(client_updates)
        
        # 服务器评估模型
        global_test_loss, global_test_acc = server.evaluate(global_testloader)
        print(f"全局模型测试结果: Loss = {global_test_loss:.4f}, Acc = {global_test_acc:.4f}")
        
        # 保存最佳模型及信息
        if global_test_acc > best_accuracy:
            best_accuracy = global_test_acc
            print(f"新的最佳准确率: {best_accuracy:.4f}, 保存模型到 {best_model_dir}")
            
            # 1. 保存模型参数
            model_state = {
                'encoder': server.get_encoder(),
                'moe_gate': server.get_moe_gate(),
                'local_expert': server.get_local_expert(),
                'pred_head': server.get_pred_head(),
                'accuracy': best_accuracy,
                'round': round + 1
            }
            torch.save(model_state, model_params_path)
            
            # 2. 准备需要保存的信息
            save_info = {
                "hyperparameters": vars(args),  # 超参数
                "global_test_result": {
                    "test_loss": float(global_test_loss),
                    "test_accuracy": float(global_test_acc),
                    "best_round": round + 1
                },
                "client_test_results": client_test_results  # 客户端测试结果
            }
            
            # 3. 保存为JSON文件
            with open(info_json_path, 'w', encoding='utf-8') as f:
                json.dump(save_info, f, ensure_ascii=False, indent=4)

def split_data(dataset, num_clients, data_split_method):
    """将数据集分割给多个客户端"""
    if data_split_method == 'iid':
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        client_datasets = []
        data_per_client = len(dataset) // num_clients
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < num_clients - 1 else len(dataset)
            client_indices = indices[start_idx:end_idx]
            client_dataset = Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
        return client_datasets
    elif data_split_method == 'non-iid':
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        sorted_indices = np.argsort(labels)
        client_datasets = []
        data_per_client = len(dataset) // num_clients
        for i in range(num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < num_clients - 1 else len(dataset)
            client_indices = sorted_indices[start_idx:end_idx]
            client_dataset = Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
        return client_datasets
    elif data_split_method == 'dirichlet':
        num_classes = len(np.unique([dataset[i][1] for i in range(len(dataset))]))
        alpha = 0.5
        label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
        class_indices = [np.where(np.array([dataset[i][1] for i in range(len(dataset))]) == c)[0] for c in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            class_idx = class_indices[c]
            np.random.shuffle(class_idx)
            proportions = np.round(label_distribution[c] * len(class_idx)).astype(int)
            proportions[-1] = len(class_idx) - np.sum(proportions[:-1])
            start_idx = 0
            for i in range(num_clients):
                end_idx = start_idx + proportions[i]
                client_indices[i].extend(class_idx[start_idx:end_idx])
                start_idx = end_idx
        client_datasets = [Subset(dataset, indices) for indices in client_indices]
        return client_datasets
    else:
        raise ValueError(f"不支持的数据划分方法: {data_split_method}")
