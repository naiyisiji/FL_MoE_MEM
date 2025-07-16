import os
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from client import Client
from server import Server
from data_loader import load_partition_data_cifar10_ssl
from models import create_encoder, create_moe_gate, create_local_expert, create_pred_head
from clustering import cluster_kmeans, init_thresholds, truth_count, estep


def set_random_seeds(seed=42):
    """设置随机种子以保证实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_two_stages(args):
    # 设置随机种子
    set_random_seeds(args.seed)
    
    # 创建工作目录和最佳模型文件夹
    work_dir = os.path.join(os.getcwd(), 'work_dir')
    best_model_dir = os.path.join(work_dir, 'best_model')
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # 加载SSL数据
    label_data_num, unlabel_data_num, test_data_num, train_data_global, test_data_global, \
    label_local_num_dict, unlabel_local_num_dict, label_local_dict, unlabel_local_dict, \
    test_data_local_dict, class_num = load_partition_data_cifar10_ssl(
        args.dataset, './data', args.data_split_method, 0.5, args.num_clients, args.batch_size, 0.1
    )

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
    for client_id in range(args.num_clients):
        # 为每个客户端创建独立模型
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
            train_data=label_local_dict[client_id].dataset,
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
    scheduler = lr_scheduler.StepLR(server.optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    server.set_lr_scheduler(scheduler)

    # 初始化最佳准确率和模型保存路径
    best_accuracy = 0.0
    model_params_path = os.path.join(best_model_dir, 'model_params.pth')
    info_json_path = os.path.join(best_model_dir, 'info.json')

    # 第一阶段：使用有标签数据训练
    print("=== 第一阶段：使用有标签数据训练 ===")
    for round in range(args.num_rounds):
        print(f"\n=== 第 {round + 1}/{args.num_rounds} 轮 ===")

        client_updates = []
        client_test_results = []

        for client_id, client in enumerate(clients):
            # 获取全局模型参数
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

            print(f"客户端 {client.client_id} 本地训练中...")
            encoder, moe_gate, local_expert, pred_head = client.train(args.local_epochs)

            # 本地评估
            local_loss, local_acc = client.evaluate(test_data_local_dict[client_id])
            print(f"客户端 {client.client_id} 本地模型测试结果: Loss = {local_loss:.4f}, Acc = {local_acc:.4f}")
            client_test_results.append({
                "client_id": client.client_id,
                "test_loss": float(local_loss),
                "test_accuracy": float(local_acc)
            })

            # 收集更新
            client_updates.append({
                'client_id': client.client_id,
                'encoder': encoder,
                'moe_gate': moe_gate,
                'local_expert': local_expert,
                'pred_head': pred_head
            })

        # 服务器聚合更新
        server.aggregate(client_updates)

        # 全局评估
        global_test_loss, global_test_acc = server.evaluate(test_data_global)
        print(f"全局模型测试结果: Loss = {global_test_loss:.4f}, Acc = {global_test_acc:.4f}")

        # 保存最佳模型
        if global_test_acc > best_accuracy:
            best_accuracy = global_test_acc
            print(f"新的最佳准确率: {best_accuracy:.4f}, 保存模型到 {best_model_dir}")

            # 收集所有客户端的模型参数
            client_models = []
            for client in clients:
                client_models.append({
                    'client_id': client.client_id,
                    'encoder': client.encoder.state_dict(),
                    'moe_gate': client.moe_gate.state_dict(),
                    'local_expert': client.local_expert.state_dict(),
                    'pred_head': client.pred_head.state_dict()
                })

            # 保存服务器和所有客户端的模型参数
            model_state = {
                'server': {
                    'encoder': server.encoder.state_dict(),
                    'moe_gate': server.moe_gate.state_dict(),
                    'local_expert': server.local_expert.state_dict(),
                    'pred_head': server.pred_head.state_dict()
                },
                'clients': client_models,
                'accuracy': best_accuracy,
                'round': round + 1
            }
            torch.save(model_state, model_params_path)

            # 保存训练信息
            save_info = {
                "hyperparameters": vars(args),
                "global_test_result": {
                    "test_loss": float(global_test_loss),
                    "test_accuracy": float(global_test_acc),
                    "best_round": round + 1
                },
                "client_test_results": client_test_results
            }

            with open(info_json_path, 'w', encoding='utf-8') as f:
                json.dump(save_info, f, ensure_ascii=False, indent=4)






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # 第二阶段：分为两部分执行
    print("\n=== 第二阶段：开始两部分训练 ===")
    # 读取第一阶段保存的最佳模型
    best_model_path = os.path.join(best_model_dir, 'model_params.pth')
    best_model = torch.load(best_model_path, map_location=args.device)
    
    # ==================== 第二阶段第一部分：聚类与标签更新 ====================
    print("\n=== 第二阶段第一部分：聚类无标签数据并更新标签数据集 ===")
    # 创建客户端ID到参数索引的映射
    client_id_to_idx = {cm['client_id']: idx for idx, cm in enumerate(best_model['clients'])}
    
    # 逐个客户端处理（依次读取参数并处理无标签数据）
    for client_id, client in enumerate(clients):
        print(f"\n--- 处理客户端 {client_id} ---")
        # 读取当前客户端的第一阶段最佳模型参数
        if client_id not in client_id_to_idx:
            print(f"警告: 未找到客户端 {client_id} 的最佳模型参数，跳过处理")
            continue
        client_params = best_model['clients'][client_id_to_idx[client_id]]
        client.encoder.load_state_dict(client_params['encoder'])
        client.moe_gate.load_state_dict(client_params['moe_gate'])
        client.local_expert.load_state_dict(client_params['local_expert'])
        client.pred_head.load_state_dict(client_params['pred_head'])
        
        # 获取当前客户端的无标签数据
        unlabel_data = unlabel_local_dict[client_id].dataset
        print(f"客户端 {client_id} 无标签数据量: {len(unlabel_data)}")
        
        # 使用encoder提取无标签数据特征，并确保特征为二维
        client.encoder.eval()
        unlabel_features = []
        with torch.no_grad():
            unlabel_loader = DataLoader(unlabel_data, batch_size=args.batch_size, shuffle=False)
            for data, _ in unlabel_loader:  # 假设无标签数据的target为占位符
                data = data.to(args.device)
                feat = client.encoder(data)
                
                # 关键修正：将高维特征展平为二维 (样本数×特征数)
                # 根据实际特征形状调整展平方式
                if len(feat.shape) > 2:
                    # 方法1: 全局平均池化 (适用于卷积特征)
                    # feat = torch.mean(feat, dim=[1, 2])  # 若特征是 (B, C, H, W)
                    
                    # 方法2: 直接展平 (适用于任何高维特征)
                    feat = feat.view(feat.size(0), -1)  # 展平为 (B, 特征数)
                
                unlabel_features.extend(feat.cpu().numpy())
        unlabel_features = np.array(unlabel_features)
        
        # 验证特征维度
        if len(unlabel_features.shape) != 2:
            raise ValueError(f"特征维度错误: 期望2维，实际{len(unlabel_features.shape)}维")
        print(f"特征形状: {unlabel_features.shape} (样本数×特征数)")
        
        # 执行K-means聚类
        cluster_datasets, cluster_fea_sets, centroids = cluster_kmeans(
            unlabel_data, unlabel_features, class_num, args.seed
        )
        for i in range(len(cluster_datasets)):
            print(f"聚类 {i} 包含样本数: {len(cluster_datasets[i])}")
        
        # 初始化阈值
        thresholds = init_thresholds(
            cluster_fea_sets, len(cluster_fea_sets), centroids, percent=args.percent
        )
        
        # 计算真实分布并保存
        df = truth_count(cluster_datasets, len(cluster_datasets))
        # 创建保存文件夹
        folder_path = (f'fedavg_kmeans/p{args.plabel}_percent{args.percent}_r'
                      f'{args.r1}_{args.r2}_{args.r3}_t{args.t1}_{args.t2}')
        os.makedirs(folder_path, exist_ok=True)
        # 保存聚类结果到Excel
        excel_path = os.path.join(folder_path, f'ground_truth_dist_r{best_model["round"]}_c{client_id}.xlsx')
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
                # 执行estep更新标签数据集
        labeled_loader = DataLoader(label_local_dict[client_id].dataset, batch_size=args.batch_size)
        mixed_dataset, _, _ = estep(
            args, 
            client.encoder, 
            args.device, 
            labeled_loader, 
            cluster_datasets, 
            cluster_fea_sets, 
            len(cluster_datasets), 
            centroids, 
            thresholds, 
            args.batch_size, 
            args.r1, 
            args.r2, 
            args.r3, 
            args.t1, 
            args.t2
        )
        
        # 更新当前客户端的有标签数据集 - 修正版本
        data_list = []
        target_list = []
        first_shape = None
        
        for item in mixed_dataset:
            # 确保每个项是(data, target)的元组
            if len(item) != 2:
                print(f"警告: 跳过格式不正确的项 {item}")
                continue
                
            data, target = item
            
            # 处理数据部分
            if not isinstance(data, torch.Tensor):
                try:
                    data = torch.tensor(data)
                except:
                    print(f"警告: 无法将数据转换为张量 {data}")
                    continue
            
            if first_shape is None:
                first_shape = data.shape
            if data.shape != first_shape:
                print(f"警告: 跳过形状不匹配的样本 {data.shape} (预期: {first_shape})")
                continue
            
            # 处理标签部分 - 关键修正
            try:
                # 尝试将标签转换为整数
                if isinstance(target, torch.Tensor):
                    # 如果是张量，确保是标量整数
                    if target.numel() != 1:
                        raise ValueError(f"标签张量元素数量不为1: {target.numel()}")
                    target = int(target.item())
                elif isinstance(target, float):
                    # 如果是浮点数，确保是整数形式（如3.0）
                    if not target.is_integer():
                        raise ValueError(f"标签浮点数不是整数: {target}")
                    target = int(target)
                elif not isinstance(target, int):
                    # 其他类型尝试直接转换为整数
                    target = int(target)
            except:
                print(f"警告: 跳过无效标签 {target} (类型: {type(target)})")
                continue
            
            data_list.append(data)
            target_list.append(target)
        
        if not data_list:
            print(f"警告: 客户端 {client_id} 没有有效的数据样本，使用原始数据集")
            continue
        
        # 堆叠所有有效数据
        data_tensor = torch.stack(data_list)
        # 确保标签是整数类型张量
        target_tensor = torch.tensor(target_list, dtype=torch.long)
        updated_dataset = torch.utils.data.TensorDataset(data_tensor, target_tensor)
        label_local_dict[client_id] = torch.utils.data.Subset(updated_dataset, range(len(updated_dataset)))
        print(f"客户端 {client_id} 标签数据集更新完成，新数据量: {len(label_local_dict[client_id])} (过滤掉 {len(mixed_dataset) - len(data_list)} 个无效样本)")

    
    # 释放最佳模型内存
    del best_model
    
    # ==================== 第二阶段第二部分：使用更新后的标签数据训练 ====================
    print("\n=== 第二阶段第二部分：使用更新后的标签数据进行训练 ===")
    # 初始化客户端训练数据为更新后的标签数据集
    for client_id, client in enumerate(clients):
        client.train_data = label_local_dict[client_id].dataset
        client.data_loader = DataLoader(
            client.train_data, 
            batch_size=args.batch_size, 
            shuffle=True
        )
    
    # 重新初始化最佳准确率和保存路径（添加MEM后缀）
    best_accuracy_mem = 0.0
    model_params_mem_path = os.path.join(best_model_dir, 'model_params_MEM.pth')
    info_json_mem_path = os.path.join(best_model_dir, 'info_MEM.json')
    
    # 训练循环（与第一阶段流程完全一致）
    for round in range(args.num_rounds):
        print(f"\n=== 第 {round + 1}/{args.num_rounds} 轮（第二阶段第二部分）===")
        client_updates = []
        client_test_results = []
        
        for client_id, client in enumerate(clients):
            # 获取全局模型参数
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
            
            print(f"客户端 {client.client_id} 本地训练中...")
            encoder, moe_gate, local_expert, pred_head = client.train(args.local_epochs)
            
            # 本地评估
            local_loss, local_acc = client.evaluate(test_data_local_dict[client_id])
            print(f"客户端 {client.client_id} 本地模型测试结果: Loss = {local_loss:.4f}, Acc = {local_acc:.4f}")
            client_test_results.append({
                "client_id": client.client_id,
                "test_loss": float(local_loss),
                "test_accuracy": float(local_acc)
            })
            
            # 收集更新
            client_updates.append({
                'client_id': client.client_id,
                'encoder': encoder,
                'moe_gate': moe_gate,
                'local_expert': local_expert,
                'pred_head': pred_head
            })
        
        # 服务器聚合更新
        server.aggregate(client_updates)
        
        # 全局评估
        global_test_loss, global_test_acc = server.evaluate(test_data_global)
        print(f"全局模型测试结果: Loss = {global_test_loss:.4f}, Acc = {global_test_acc:.4f}")
        
        # 保存最佳模型（带MEM后缀）
        if global_test_acc > best_accuracy_mem:
            best_accuracy_mem = global_test_acc
            print(f"新的最佳准确率 (MEM): {best_accuracy_mem:.4f}, 保存模型到 {best_model_dir}")
            
            # 收集所有客户端的模型参数
            client_models = []
            for client in clients:
                client_models.append({
                    'client_id': client.client_id,
                    'encoder': client.encoder.state_dict(),
                    'moe_gate': client.moe_gate.state_dict(),
                    'local_expert': client.local_expert.state_dict(),
                    'pred_head': client.pred_head.state_dict()
                })
            
            # 保存服务器和所有客户端的模型参数
            model_state = {
                'server': {
                    'encoder': server.encoder.state_dict(),
                    'moe_gate': server.moe_gate.state_dict(),
                    'local_expert': server.local_expert.state_dict(),
                    'pred_head': server.pred_head.state_dict()
                },
                'clients': client_models,
                'accuracy': best_accuracy_mem,
                'round': round + 1
            }
            torch.save(model_state, model_params_mem_path)
            
            # 保存训练信息
            save_info = {
                "hyperparameters": vars(args),
                "global_test_result": {
                    "test_loss": float(global_test_loss),
                    "test_accuracy": float(global_test_acc),
                    "best_round": round + 1
                },
                "client_test_results": client_test_results
            }
            
            with open(info_json_mem_path, 'w', encoding='utf-8') as f:
                json.dump(save_info, f, ensure_ascii=False, indent=4)
        
        # 更新学习率
        if server.lr_scheduler:
            server.lr_scheduler.step()
            print(f"全局学习率已更新为: {server.optimizer.param_groups[0]['lr']:.6f}")


    print("\n两阶段训练完成！")
    return server
