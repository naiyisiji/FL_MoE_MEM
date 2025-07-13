import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from models import create_local_expert

class Client:
    def __init__(self, client_id, encoder, moe_gate, local_expert, pred_head, train_data, batch_size, device, use_moe=False, a_weight=0.5, b_weight=0.5, top_k=2, lr=0.001):
        self.client_id = client_id
        self.encoder = encoder
        self.moe_gate = moe_gate
        self.local_expert = local_expert
        self.pred_head = pred_head
        self.train_data = train_data
        self.batch_size = batch_size
        self.device = device
        self.use_moe = use_moe
        self.a_weight = a_weight
        self.b_weight = b_weight
        self.top_k = top_k
        self.lr = lr

        # 初始化专家和独热向量
        self.experts = None
        self.one_hot_vector = None

        # 用于计算loss
        self.criterion = nn.CrossEntropyLoss()

    def set_encoder(self, encoder_state_dict):
        self.encoder.load_state_dict(encoder_state_dict)

    def set_moe_gate(self, moe_gate_state_dict):
        self.moe_gate.load_state_dict(moe_gate_state_dict)

    def set_local_expert(self, local_expert_state_dict):
        self.local_expert.load_state_dict(local_expert_state_dict)

    def set_pred_head(self, pred_head_state_dict):
        self.pred_head.load_state_dict(pred_head_state_dict)

    def set_experts(self, experts, one_hot_vector):
        self.experts = experts
        self.one_hot_vector = one_hot_vector

    def train(self, local_epochs):
        # 创建数据加载器
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        # 定义优化器
        optimizer = optim.Adam(list(self.encoder.parameters()) + 
                              list(self.local_expert.parameters()) + 
                              list(self.pred_head.parameters()), 
                              lr=self.lr)

        # 如果使用moe且接收到了其他专家，则也训练moe_gate
        if self.use_moe and self.experts is not None:
            optimizer.add_param_group({'params': self.moe_gate.parameters()})

        self.encoder.train()
        self.moe_gate.train()
        self.local_expert.train()
        self.pred_head.train()

        for epoch in range(local_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 清空梯度
                optimizer.zero_grad()

                # 分支1: encoder -> local_expert -> pred_head
                features = self.encoder(inputs)
                local_output = self.local_expert(features)
                local_pred = self.pred_head(local_output)
                loss1 = self.criterion(local_pred, labels)

                if self.use_moe and self.experts is not None:
                    # 分支2: encoder -> moe_gate -> experts -> pred_head
                    gate_features = nn.Flatten()(features)  # 展平用于门控
                    gate_output = self.moe_gate(gate_features)

                    # 保存topk个分数，其他置零并归一化
                    top_k_values, top_k_indices = torch.topk(gate_output, self.top_k, dim=1)
                    zeroed_gate_output = torch.zeros_like(gate_output)
                    batch_size = gate_output.size(0)
                    for i in range(batch_size):
                        zeroed_gate_output[i, top_k_indices[i]] = top_k_values[i]
                    normalized_gate_output = torch.softmax(zeroed_gate_output, dim=1)

                    # 将本地专家嵌入到指定位置
                    one_hot_index = np.argmax(self.one_hot_vector)
                    all_experts = self.experts.copy()
                    
                    # 确保门控输出维度与专家数量一致
                    # 门控输出维度应为客户端总数(包括本地客户端)
                    expected_num_experts = len(self.one_hot_vector)
                    
                    # 如果专家列表长度不足，填充None
                    while len(all_experts) < expected_num_experts - 1:
                        all_experts.append(None)
                    
                    # 插入本地专家
                    if one_hot_index < len(all_experts):
                        all_experts.insert(one_hot_index, self.local_expert.state_dict())
                    else:
                        # 如果one_hot_index超出范围，将本地专家添加到末尾
                        all_experts.append(self.local_expert.state_dict())
                    
                    # 扩展独热向量以匹配批量大小
                    target = torch.tensor(self.one_hot_vector, dtype=torch.float32).to(self.device)
                    target = target.unsqueeze(0).repeat(gate_output.size(0), 1)

                    # 将独热编码转换为类别索引
                    target_indices = torch.argmax(target, dim=1).long()

                    # 计算门控向量和独热向量的损失
                    loss_gate = self.criterion(normalized_gate_output, target_indices)

                    # 根据门控向量选择所有专家
                    expert_outputs = []
                    for batch_idx in range(features.size(0)):
                        batch_expert_output = 0
                        valid_experts = 0
                        
                        # 确保专家循环不超过门控输出维度
                        for expert_index in range(min(len(all_experts), normalized_gate_output.size(1))):
                            expert_state = all_experts[expert_index]  # 获取当前专家状态
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
                                valid_experts += 1
                        
                        # 如果没有有效的专家，使用本地专家作为后备
                        if valid_experts == 0:
                            batch_expert_output = self.local_expert(features[batch_idx].unsqueeze(0))
                        else:
                            batch_expert_output /= valid_experts
                        
                        expert_outputs.append(batch_expert_output)

                    expert_outputs = torch.cat(expert_outputs, dim=0)
                    expert_pred = self.pred_head(expert_outputs)
                    loss_expert = self.criterion(expert_pred, labels)

                    # 分支2总损失
                    loss2 = self.a_weight * loss_gate + self.b_weight * loss_expert

                    # 总损失
                    loss = loss1 + loss2

                    # 计算准确率
                    _, predicted = expert_pred.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    # 只更新moe_gate的梯度
                    for param in self.encoder.parameters():
                        param.grad = None
                    for param in self.local_expert.parameters():
                        param.grad = None
                    for param in self.pred_head.parameters():
                        param.grad = None
                else:
                    # 不使用moe或没有接收到其他专家
                    loss = loss1

                    # 计算准确率
                    _, predicted = local_pred.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 打印本轮训练结果
            print(f"客户端 {self.client_id}, 轮次 {epoch+1}/{local_epochs}, 损失: {total_loss/len(train_loader):.4f}, 准确率: {100.*correct/total:.2f}%")

        # 返回更新后的模型参数
        return (self.encoder.state_dict(), 
                self.moe_gate.state_dict(), 
                self.local_expert.state_dict(), 
                self.pred_head.state_dict())

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
                if self.use_moe and self.experts is not None:
                    features = self.encoder(inputs)
                    gate_features = nn.Flatten()(features)
                    gate_output = self.moe_gate(gate_features)

                    # 保存topk个分数，其他置零并归一化
                    top_k_values, top_k_indices = torch.topk(gate_output, self.top_k, dim=1)
                    zeroed_gate_output = torch.zeros_like(gate_output)
                    batch_size = gate_output.size(0)
                    for i in range(batch_size):
                        zeroed_gate_output[i, top_k_indices[i]] = top_k_values[i]
                    normalized_gate_output = torch.softmax(zeroed_gate_output, dim=1)

                    # 将本地专家嵌入到指定位置
                    one_hot_index = np.argmax(self.one_hot_vector)
                    all_experts = self.experts.copy()
                    
                    # 确保门控输出维度与专家数量一致
                    # 门控输出维度应为客户端总数(包括本地客户端)
                    expected_num_experts = len(self.one_hot_vector)
                    
                    # 如果专家列表长度不足，填充None
                    while len(all_experts) < expected_num_experts - 1:
                        all_experts.append(None)
                    
                    # 插入本地专家
                    if one_hot_index < len(all_experts):
                        all_experts.insert(one_hot_index, self.local_expert.state_dict())
                    else:
                        # 如果one_hot_index超出范围，将本地专家添加到末尾
                        all_experts.append(self.local_expert.state_dict())
                    
                    # 根据门控向量选择所有专家
                    expert_outputs = []
                    for batch_idx in range(features.size(0)):
                        batch_expert_output = 0
                        valid_experts = 0
                        
                        # 确保专家循环不超过门控输出维度
                        for expert_index in range(min(len(all_experts), normalized_gate_output.size(1))):
                            expert_state = all_experts[expert_index]  # 获取当前专家状态
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
                                valid_experts += 1
                        
                        # 如果没有有效的专家，使用本地专家
                        if valid_experts == 0:
                            batch_expert_output = self.local_expert(features[batch_idx].unsqueeze(0))
                        else:
                            batch_expert_output /= valid_experts
                        
                        expert_outputs.append(batch_expert_output)
                    
                    expert_outputs = torch.cat(expert_outputs, dim=0)
                    outputs = self.pred_head(expert_outputs)
                else:
                    # 不使用moe
                    features = self.encoder(inputs)
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
