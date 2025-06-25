import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Callable
from torchvision.models import resnet18, resnet34, resnet50, ResNet  # 导入具体的ResNet模型和基类

class Client:
    def __init__(self, client_id: int, train_data: Dataset, test_data: Dataset, 
                 backbone: nn.Module, decoder: nn.Module, gate: nn.Module,
                 device_ids: List[int], 
                 batch_size: int = 32, lr: float = 0.01, 
                 epochs: int = 5, optimizer: str = "sgd", 
                 criterion: Callable = nn.CrossEntropyLoss(),
                 use_moe=False, moe_strategy='single_expert'):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.device_ids = device_ids
        self.batch_size = max(2, batch_size)
        self.lr = lr
        self.epochs = epochs
        self.criterion = criterion
        self.use_moe = use_moe
        self.moe_strategy = moe_strategy
        self.first_global_round_completed = False  # 新增：标记第一次全局训练是否完成

        # 使用DataParallel在多个GPU上并行训练
        self.backbone = nn.DataParallel(backbone, device_ids=device_ids)
        self.backbone.to(f"cuda:{device_ids[0]}")
        self.decoder = nn.DataParallel(decoder, device_ids=device_ids)
        self.decoder.to(f"cuda:{device_ids[0]}")
        self.gate = nn.DataParallel(gate, device_ids=device_ids)
        self.gate.to(f"cuda:{device_ids[0]}")

        # 根据指定的优化器名称创建优化器
        if optimizer.lower() == "sgd":
            self.optimizer_backbone = optim.SGD(self.backbone.parameters(), lr=lr, momentum=0.9)
            self.optimizer_decoder = optim.SGD(self.decoder.parameters(), lr=lr, momentum=0.9)
            self.optimizer_gate = optim.SGD(self.gate.parameters(), lr=lr, momentum=0.9)
        elif optimizer.lower() == "adam":
            self.optimizer_backbone = optim.Adam(self.backbone.parameters(), lr=lr)
            self.optimizer_decoder = optim.Adam(self.decoder.parameters(), lr=lr)
            self.optimizer_gate = optim.Adam(self.gate.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer}")

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(test_data, batch_size=batch_size)

        if use_moe:
            if moe_strategy == 'single_expert':
                self.expert_decoder = None
            elif moe_strategy == 'multi_expert':
                self.expert_decoders = []

    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """在本地数据集上训练模型，并返回模型参数"""
        self.backbone.train()
        self.decoder.train()
        self.gate.train()
        
        # 第一次全局训练完成后才使用专家模型
        if self.use_moe and self.first_global_round_completed:
            if self.moe_strategy == 'single_expert' and self.expert_decoder is not None:
                self.expert_decoder.eval()  # 专家模型不训练
            elif self.moe_strategy == 'multi_expert' and self.expert_decoders:
                for decoder in self.expert_decoders:
                    decoder.eval()  # 专家模型不训练

        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(f"cuda:{self.device_ids[0]}"), labels.to(f"cuda:{self.device_ids[0]}")
                
                self.optimizer_backbone.zero_grad()
                self.optimizer_decoder.zero_grad()
                self.optimizer_gate.zero_grad()

                features = self.backbone(inputs)
                features = features.view(features.size(0), -1)
                outputs = self.decoder(features)
                
                if self.use_moe and self.first_global_round_completed:
                    gate_output = torch.sigmoid(self.gate(features))

                    if self.moe_strategy == 'single_expert' and self.expert_decoder is not None:
                        expert_outputs = self.expert_decoder(features)
                        combined_outputs = gate_output * outputs + (1 - gate_output) * expert_outputs
                    elif self.moe_strategy == 'multi_expert' and self.expert_decoders:
                        expert_outputs = []
                        for decoder in self.expert_decoders:
                            expert_outputs.append(decoder(features))
                        expert_outputs = torch.stack(expert_outputs, dim=0)
                        gate_weights = torch.softmax(self.gate(features).squeeze(-1), dim=0)
                        weighted_expert_outputs = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=0)
                        combined_outputs = gate_output * outputs + (1 - gate_output) * weighted_expert_outputs
                    final_outputs = combined_outputs
                else:
                    # 非MoE模式下直接使用decoder输出
                    final_outputs = outputs

                loss = self.criterion(final_outputs, labels)
                loss.backward()
                self.optimizer_backbone.step()
                self.optimizer_decoder.step()
                self.optimizer_gate.step()
                
                total_loss += loss.item()
                _, predicted = final_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            print(f"客户端 {self.client_id}, 轮次 {epoch+1}/{self.epochs}, "
                  f"损失: {total_loss/len(self.train_loader):.4f}, "
                  f"准确率: {100.*correct/total:.2f}%")
        
        backbone_state_dict = self.backbone.state_dict()
        decoder_state_dict = self.decoder.state_dict()
        return {k: v.cpu().detach() for k, v in backbone_state_dict.items()}, {k: v.cpu().detach() for k, v in decoder_state_dict.items()}

    def evaluate(self) -> Tuple[float, float]:
        """在测试集上评估模型"""
        self.backbone.eval()
        self.decoder.eval()
        self.gate.eval()
        
        # 第一次全局训练完成后才使用专家模型
        if self.use_moe and self.first_global_round_completed:
            if self.moe_strategy == 'single_expert' and self.expert_decoder is not None:
                self.expert_decoder.eval()
            elif self.moe_strategy == 'multi_expert' and self.expert_decoders:
                for decoder in self.expert_decoders:
                    decoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(f"cuda:{self.device_ids[0]}"), labels.to(f"cuda:{self.device_ids[0]}")
                features = self.backbone(inputs)
                features = features.view(features.size(0), -1)
                outputs = self.decoder(features)
                
                if self.use_moe and self.first_global_round_completed:
                    gate_output = torch.sigmoid(self.gate(features))

                    if self.moe_strategy == 'single_expert' and self.expert_decoder is not None:
                        expert_outputs = self.expert_decoder(features)
                        combined_outputs = gate_output * outputs + (1 - gate_output) * expert_outputs
                    elif self.moe_strategy == 'multi_expert' and self.expert_decoders:
                        expert_outputs = []
                        for decoder in self.expert_decoders:
                            expert_outputs.append(decoder(features))
                        expert_outputs = torch.stack(expert_outputs, dim=0)
                        gate_weights = torch.softmax(self.gate(features).squeeze(-1), dim=0)
                        weighted_expert_outputs = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=0)
                        combined_outputs = gate_output * outputs + (1 - gate_output) * weighted_expert_outputs
                    final_outputs = combined_outputs
                else:
                    # 非MoE模式下直接使用decoder输出
                    final_outputs = outputs

                loss = self.criterion(final_outputs, labels)
                
                total_loss += loss.item()
                _, predicted = final_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        print(f"客户端 {self.client_id} 测试结果: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.2f}%")
        return avg_loss, accuracy

    def update_expert_model(self, expert_decoder_params):
        if self.use_moe:
            if self.moe_strategy == 'single_expert':
                if self.expert_decoder is None:
                    # 获取原始模型
                    base_decoder = self.decoder.module
                    
                    # 检查模型类型并创建新模型
                    if isinstance(base_decoder, nn.Linear):
                        num_classes = base_decoder.out_features
                        input_features = base_decoder.in_features
                        new_decoder = nn.Linear(input_features, num_classes)
                        self.expert_decoder = nn.DataParallel(new_decoder, device_ids=self.device_ids)
                        self.expert_decoder.to(f"cuda:{self.device_ids[0]}")
                
                # 加载专家模型参数
                expert_decoder_state = self.expert_decoder.state_dict()
                for name, param in expert_decoder_params.items():
                    if name in expert_decoder_state:
                        expert_decoder_state[name] = param
                self.expert_decoder.load_state_dict(expert_decoder_state)
            
            elif self.moe_strategy == 'multi_expert':
                self.expert_decoders = []
                base_decoder = self.decoder.module
                
                for _ in range(len(expert_decoder_params)):
                    # 检查模型类型并创建新模型
                    if isinstance(base_decoder, nn.Linear):
                        num_classes = base_decoder.out_features
                        input_features = base_decoder.in_features
                        new_decoder = nn.Linear(input_features, num_classes)
                        expert_decoder = nn.DataParallel(new_decoder, device_ids=self.device_ids)
                        expert_decoder.to(f"cuda:{self.device_ids[0]}")
                        self.expert_decoders.append(expert_decoder)
                
                # 加载每个专家模型的参数
                for i, decoder in enumerate(self.expert_decoders):
                    decoder_state = decoder.state_dict()
                    for name, param in expert_decoder_params[i].items():
                        if name in decoder_state:
                            decoder_state[name] = param
                    decoder.load_state_dict(decoder_state)
