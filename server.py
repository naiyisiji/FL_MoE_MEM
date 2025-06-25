import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Callable, Optional

class Server:
    def __init__(self, backbone: nn.Module, decoder: nn.Module, device_ids: List[int], 
                 aggregation_method: str = "fedavg", use_moe=False, moe_strategy='single_expert'):
        self.device_ids = device_ids
        self.global_backbone = nn.DataParallel(backbone, device_ids=device_ids)
        self.global_backbone.to(f"cuda:{device_ids[0]}")
        self.global_decoder = nn.DataParallel(decoder, device_ids=device_ids)
        self.global_decoder.to(f"cuda:{device_ids[0]}")
        self.aggregation_method = aggregation_method
        self.client_weights = None
        self.use_moe = use_moe
        self.moe_strategy = moe_strategy
        
        # 新增：用于评估的聚合decoder
        self.evaluation_decoder = nn.DataParallel(decoder, device_ids=device_ids)
        self.evaluation_decoder.to(f"cuda:{device_ids[0]}")
        
        if use_moe:
            if moe_strategy == 'single_expert':
                self.expert_decoder = nn.DataParallel(decoder, device_ids=device_ids)
                self.expert_decoder.to(f"cuda:{device_ids[0]}")
            elif moe_strategy == 'multi_expert':
                self.expert_decoders = []

    def aggregate(self, client_backbone_params: List[Dict[str, torch.Tensor]], 
                  client_decoder_params: List[Dict[str, torch.Tensor]],
                  client_sizes: Optional[List[int]] = None) -> None:
        """聚合客户端模型参数"""
        if not client_backbone_params:
            print("警告: 没有收到任何客户端参数，跳过本轮聚合")
            return
            
        if self.aggregation_method.lower() == "fedavg":
            # 计算聚合权重
            if client_sizes is not None and sum(client_sizes) > 0:
                total_size = sum(client_sizes)
                weights = [size / total_size for size in client_sizes]
            else:
                weights = [1.0 / len(client_backbone_params) for _ in client_backbone_params]
            
            # 聚合backbone参数
            aggregated_backbone_params = {}
            first_client_backbone_params = client_backbone_params[0]
            for name in first_client_backbone_params.keys():
                if name in self.global_backbone.state_dict():
                    aggregated_backbone_params[name] = weights[0] * first_client_backbone_params[name].clone()
            for i in range(1, len(client_backbone_params)):
                client_backbone_state = client_backbone_params[i]
                for name in client_backbone_state.keys():
                    if name in self.global_backbone.state_dict():
                        aggregated_backbone_params[name] += weights[i] * client_backbone_state[name]
            server_backbone_state = self.global_backbone.state_dict()
            updated_backbone_state = {}
            for name in server_backbone_state.keys():
                if name in aggregated_backbone_params:
                    updated_backbone_state[name] = aggregated_backbone_params[name]
                else:
                    updated_backbone_state[name] = server_backbone_state[name]
            self.global_backbone.load_state_dict(updated_backbone_state)

            # 聚合decoder参数到evaluation_decoder
            aggregated_decoder_params = {}
            first_client_decoder_params = client_decoder_params[0]
            for name in first_client_decoder_params.keys():
                if name in self.evaluation_decoder.state_dict():
                    aggregated_decoder_params[name] = weights[0] * first_client_decoder_params[name].clone()
            for i in range(1, len(client_decoder_params)):
                client_decoder_state = client_decoder_params[i]
                for name in client_decoder_state.keys():
                    if name in self.evaluation_decoder.state_dict():
                        aggregated_decoder_params[name] += weights[i] * client_decoder_state[name]
            evaluation_decoder_state = self.evaluation_decoder.state_dict()
            for name in evaluation_decoder_state.keys():
                if name in aggregated_decoder_params:
                    evaluation_decoder_state[name] = aggregated_decoder_params[name]
            self.evaluation_decoder.load_state_dict(evaluation_decoder_state)

            # 处理decoder参数（用于MoE）
            if self.use_moe:
                if self.moe_strategy == 'single_expert':
                    # 简单平均聚合decoder作为单一专家模型
                    expert_decoder_state = self.expert_decoder.state_dict()
                    for name in expert_decoder_state.keys():
                        if name in aggregated_decoder_params:
                            expert_decoder_state[name] = aggregated_decoder_params[name]
                    self.expert_decoder.load_state_dict(expert_decoder_state)
                elif self.moe_strategy == 'multi_expert':
                    # 获取输入和输出维度
                    base_decoder = self.global_decoder.module
                    if isinstance(base_decoder, nn.Linear):
                        in_features = base_decoder.in_features
                        out_features = base_decoder.out_features
                    else:
                        try:
                            in_features = base_decoder[0].in_features
                            out_features = base_decoder[-1].out_features
                        except:
                            raise ValueError("无法确定decoder的输入输出维度")
                    
                    # 创建多个专家decoder
                    self.expert_decoders = []
                    for _ in range(len(client_decoder_params)):
                        if isinstance(base_decoder, nn.Linear):
                            new_decoder = nn.Linear(in_features, out_features)
                        else:
                            new_decoder = type(base_decoder)(in_features, out_features)
                        expert_decoder = nn.DataParallel(new_decoder, device_ids=self.device_ids)
                        expert_decoder.to(f"cuda:{self.device_ids[0]}")
                        self.expert_decoders.append(expert_decoder)
                    
                    # 加载每个专家模型的参数
                    for i, decoder in enumerate(self.expert_decoders):
                        decoder_state = decoder.state_dict()
                        for name, param in client_decoder_params[i].items():
                            if name in decoder_state:
                                decoder_state[name] = param
                        decoder.load_state_dict(decoder_state)
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")

    def evaluate(self, test_loader: DataLoader, criterion: Callable = nn.CrossEntropyLoss()) -> Tuple[float, float]:
        """在全局测试集上评估模型"""
        self.global_backbone.eval()
        self.evaluation_decoder.eval()  # 使用聚合后的decoder进行评估
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(f"cuda:{self.device_ids[0]}"), labels.to(f"cuda:{self.device_ids[0]}")
                features = self.global_backbone(inputs)
                features = features.view(features.size(0), -1)
                outputs = self.evaluation_decoder(features)  # 使用evaluation_decoder
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        print(f"全局模型测试结果: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.2f}%")
        return avg_loss, accuracy

    def broadcast(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """广播全局模型参数给客户端"""
        global_backbone_params = {k: v.cpu().detach() for k, v in self.global_backbone.state_dict().items()}
        if self.use_moe:
            if self.moe_strategy == 'single_expert':
                expert_decoder_params = {k: v.cpu().detach() for k, v in self.expert_decoder.state_dict().items()}
            elif self.moe_strategy == 'multi_expert':
                expert_decoder_params = []
                for decoder in self.expert_decoders:
                    expert_decoder_params.append({k: v.cpu().detach() for k, v in decoder.state_dict().items()})
            return {'global_backbone': global_backbone_params, 'expert_decoder': expert_decoder_params}
        return {'global_backbone': global_backbone_params}

    def evaluate_(self, test_loader: DataLoader, criterion: Callable = nn.CrossEntropyLoss()) -> Tuple[float, float]:
        """在全局测试集上评估模型"""
        self.global_backbone.eval()
        self.global_decoder.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(f"cuda:{self.device_ids[0]}"), labels.to(f"cuda:{self.device_ids[0]}")
                features = self.global_backbone(inputs)
                features = features.view(features.size(0), -1)
                outputs = self.global_decoder(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        print(f"全局模型测试结果: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.2f}%")
        return avg_loss, accuracy
