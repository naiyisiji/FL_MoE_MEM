import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models import create_local_expert

class Server:
    def __init__(self, encoder, moe_gate, local_expert, pred_head, num_clients, 
                 encoder_agg_method, moe_agg_method, pred_head_agg_method, device, 
                 use_moe=False, top_k=2):  # 添加 top_k 参数
        self.encoder = encoder
        self.moe_gate = moe_gate
        self.local_expert = local_expert
        self.pred_head = pred_head
        self.num_clients = num_clients
        self.encoder_agg_method = encoder_agg_method
        self.moe_agg_method = moe_agg_method
        self.pred_head_agg_method = pred_head_agg_method
        self.device = device
        self.use_moe = use_moe
        self.top_k = top_k  # 初始化 top_k 属性
        
        # 为每个客户端创建独热向量
        self.one_hot_vectors = []
        for i in range(num_clients):
            one_hot = np.zeros(num_clients)
            one_hot[i] = 1.0
            self.one_hot_vectors.append(one_hot)
        
        # 存储所有客户端的专家模型
        self.all_experts = [None] * num_clients
        
        # 优化器和学习率调度器
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.moe_gate.parameters()) + 
            list(self.local_expert.parameters()) + 
            list(self.pred_head.parameters())
        )
        
        # 初始化学习率调度器为None，将在train.py中设置
        self.lr_scheduler = None
        
        # 用于评估的损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def set_lr_scheduler(self, scheduler):
        """设置学习率调度器"""
        self.lr_scheduler = scheduler
    
    def get_encoder(self):
        return deepcopy(self.encoder.state_dict())
    
    def get_moe_gate(self):
        return deepcopy(self.moe_gate.state_dict())
    
    def get_local_expert(self):
        return deepcopy(self.local_expert.state_dict())
    
    def get_pred_head(self):
        return deepcopy(self.pred_head.state_dict())
    
    def get_experts_and_vector(self, client_id):
        """获取所有专家(除了当前客户端自己的)和对应的独热向量"""
        # 复制所有专家
        experts = self.all_experts.copy()
        # 移除当前客户端的专家(将由客户端自己提供)
        experts[client_id] = None
        return experts, self.one_hot_vectors[client_id]
    
    def aggregate(self, client_updates):
        """聚合来自客户端的更新"""
        # 提取所有客户端的更新
        encoder_updates = [update['encoder'] for update in client_updates]
        moe_gate_updates = [update['moe_gate'] for update in client_updates]
        local_expert_updates = [update['local_expert'] for update in client_updates]
        pred_head_updates = [update['pred_head'] for update in client_updates]
        
        # 聚合编码器
        if self.encoder_agg_method == 'FedAvg':
            self._fedavg_aggregate(self.encoder, encoder_updates)
        
        # 聚合MOE门控
        if self.use_moe and self.moe_agg_method == 'FedAvg':
            self._fedavg_aggregate(self.moe_gate, moe_gate_updates)
        
        # 聚合预测头
        if self.pred_head_agg_method == 'FedAvg':
            self._fedavg_aggregate(self.pred_head, pred_head_updates)
        
        # 保存所有客户端的专家模型
        for update in client_updates:
            client_id = update['client_id']
            self.all_experts[client_id] = update['local_expert']
        
        # 聚合本地专家(仅当不使用moe时)
        if not self.use_moe:
            self._fedavg_aggregate(self.local_expert, local_expert_updates)
        
        # 更新学习率
        if self.lr_scheduler:
            self.lr_scheduler.step()
            print(f"全局学习率已更新为: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _fedavg_aggregate(self, global_model, client_models):
        """使用FedAvg方法聚合模型参数"""
        # 获取全局模型的状态字典
        global_dict = global_model.state_dict()
        
        # 初始化聚合后的参数
        for key in global_dict.keys():
            if global_dict[key].dtype in [torch.float32, torch.float64]:  # 只处理浮点类型的参数
                global_dict[key] = torch.zeros_like(global_dict[key])
        
        # 聚合所有客户端的参数
        for client_dict in client_models:
            for key in global_dict.keys():
                if global_dict[key].dtype in [torch.float32, torch.float64]:  # 只处理浮点类型的参数
                    global_dict[key] += client_dict[key] / len(client_models)
        
        # 更新全局模型
        global_model.load_state_dict(global_dict)

    
    def evaluate(self, test_loader):
        """在测试集上评估模型"""
        self.encoder.eval()
        self.moe_gate.eval()
        self.local_expert.eval()
        self.pred_head.eval()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                features = self.encoder(inputs)
                
                if self.use_moe:
                    # 使用moe门控选择专家
                    gate_features = nn.Flatten()(features)
                    gate_output = self.moe_gate(gate_features)

                    # 保存topk个分数，其他置零并归一化
                    top_k_values, top_k_indices = torch.topk(gate_output, self.top_k, dim=1)
                    zeroed_gate_output = torch.zeros_like(gate_output)
                    batch_size = gate_output.size(0)
                    for i in range(batch_size):
                        zeroed_gate_output[i, top_k_indices[i]] = top_k_values[i]
                    normalized_gate_output = torch.softmax(zeroed_gate_output, dim=1)
                    
                    expert_outputs = []
                    for batch_idx in range(features.size(0)):
                        batch_expert_output = 0
                        
                        for expert_index, expert_state in enumerate(self.all_experts):
                            if expert_state is not None and normalized_gate_output[batch_idx, expert_index] > 0:
                                # 创建专家模型实例
                                expert_input_dim = features.size(1)
                                expert_output_dim = expert_input_dim
                                expert = create_local_expert(
                                    'linear', 
                                    expert_input_dim, 
                                    expert_output_dim
                                ).to(self.device)
                                expert.load_state_dict(expert_state)
                                expert_output = expert(features[batch_idx].unsqueeze(0))
                                batch_expert_output += expert_output * normalized_gate_output[batch_idx, expert_index].unsqueeze(0).unsqueeze(1)
                        
                        # 如果没有有效的专家，使用全局专家
                        if batch_expert_output is 0:
                            batch_expert_output = self.local_expert(features[batch_idx].unsqueeze(0))
                        
                        expert_outputs.append(batch_expert_output)
                    
                    expert_outputs = torch.cat(expert_outputs, dim=0)
                    outputs = self.pred_head(expert_outputs)
                else:
                    # 不使用moe
                    local_output = self.local_expert(features)
                    outputs = self.pred_head(local_output)
                
                # 计算损失和准确率
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 计算平均损失和准确率
        avg_loss = test_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        # 将模型设置回训练模式
        self.encoder.train()
        self.moe_gate.train()
        self.local_expert.train()
        self.pred_head.train()
        
        return avg_loss, accuracy
