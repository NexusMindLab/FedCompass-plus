# FL.py (修改聚合策略为FedNova)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
import flwr as fl
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitIns, EvaluateIns


def client_training(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float,
                    device: torch.device, cid: int, server_round: int, result_dir):
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    loss_history = []
    acc_history = []
    
    # 记录本地更新次数
    local_steps = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # 记录本地更新次数
            local_steps += 1
        
        # 计算平均损失和准确率
        epoch_loss /= len(train_loader) if len(train_loader) > 0 else 1
        epoch_acc = correct / total if total > 0 else 0
        epoch_acc_percent = epoch_acc * 100
        
        # 更新学习率
        scheduler.step(epoch_loss)
        
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc_percent)
        
        # 获取当前时间戳
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 日志输出
        log_message = (
            f"[{current_time}] [Client {cid}, Round {server_round}] "
            f"Train Epoch: {epoch+1}/{epochs} \t "
            f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc_percent:.2f}%"
        )
        print(log_message, flush=True)
        
        # 写入文件
        with open(f"{result_dir}/train/client_{cid}_log.txt", "a") as f:
            f.write(
                f"[{current_time}] Round {server_round}\t Epoch {epoch + 1}\t "
                f"Loss: {epoch_loss:.4f}\tAccuracy: {epoch_acc_percent:.4f}%\n"
            )
    
    return loss_history, acc_history, model.state_dict(), local_steps

def client_evaluation(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images)
            
            # 计算损失
            loss += criterion(outputs, labels.long()).item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    loss /= len(test_loader) if len(test_loader) > 0 else 1
    accuracy = correct / total if total > 0 else 0
    
    return loss, accuracy

def server_evaluate(server_round, parameters, model: nn.Module, test_loader: DataLoader, device: torch.device, result_dir):
    # 将 Parameters 对象转为 List[np.ndarray]
    parameters = parameters_to_ndarrays(parameters)

    # 设置全局模型参数
    state_dict = {}
    for k, v in zip(model.state_dict().keys(), parameters):
        if "quantum" in k.lower():
            state_dict[k] = torch.tensor(v, dtype=torch.float64)
        else:
            state_dict[k] = torch.tensor(v, dtype=torch.float32)
            
    model.load_state_dict(state_dict)
    
    # 在服务器测试集上评估
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            
            outputs = model(images)
            
            # 计算损失
            loss += criterion(outputs, labels.long()).item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    loss /= len(test_loader) if len(test_loader) > 0 else 1
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nServer round {server_round}: Evaluation loss {loss:.4f} / accuracy {accuracy*100:.4f} %")
    
    with open(f"{result_dir}/test/server_test.txt", "a") as f:
        f.write(f"Round {server_round}\tLoss: {loss:.4f}\tAccuracy: {accuracy*100:.4f}%\n")
    
    return loss, {"accuracy": accuracy}

class HybridClient(fl.client.NumPyClient):
    def __init__(self, cid: int, train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, device: torch.device, config: Dict):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.config = config
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        state_dict = {}
        for k, v in zip(self.model.state_dict().keys(), parameters):
            if "quantum" in k.lower():
                state_dict[k] = torch.tensor(v, dtype=torch.float64)
            else:
                state_dict[k] = torch.tensor(v, dtype=torch.float32)
                
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # 从配置中获取参数
        server_round = config.get("server_round", 1)
        local_epochs = config.get("local_epochs", self.config["local_epochs"])
        learning_rate = config.get("learning_rate", self.config["learning_rate"])
        
        # 客户端训练
        print(f"[Client {self.cid}] Starting local training for round {server_round}")
        loss_hist, acc_hist, model_state, local_steps = client_training(
            self.model, 
            self.train_loader, 
            local_epochs, 
            learning_rate, 
            self.device,
            self.cid,
            server_round,
            self.config["result_dir"]
        )
        
        # 更新本地模型
        self.model.load_state_dict(model_state)
        
        # 返回更新后的参数和结果
        return self.get_parameters({}), len(self.train_loader.dataset), {
            "loss": loss_hist[-1],
            "accuracy": acc_hist[-1],
            "local_steps": local_steps  # 添加本地更新次数
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = client_evaluation(
            self.model, 
            self.test_loader, 
            self.device
        )
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

class FedNovaStrategy(fl.server.strategy.Strategy):
    """FedNova聚合策略实现（兼容不同 Flower 版本）"""

    def __init__(
        self,
        initial_parameters,
        evaluate_fn,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
    ) -> None:
        super().__init__()
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """返回 [(client_proxy, FitIns), ...] 或 []"""
        # 构造 config dict
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # 包装为 FitIns
        fit_ins = FitIns(parameters, config)

        # 可用客户端数
        available = client_manager.num_available()
        if available == 0:
            return []

        # 计算采样数量（兼顾 fraction 和 min）
        sample_size = int(self.fraction_fit * available)
        sample_size = max(sample_size, self.min_fit_clients)
        sample_size = min(sample_size, available)

        # 确保 min_num_clients 不大于 sample_size（client_manager.sample 要求）
        min_num = min(self.min_fit_clients, sample_size)

        # 采样客户端（使用位置参数以兼容不同 Flower 版本）
        clients = client_manager.sample(sample_size, min_num)

        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """返回 [(client_proxy, EvaluateIns), ...] 或 []"""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        evaluate_ins = EvaluateIns(parameters, config)

        available = client_manager.num_available()
        if available == 0:
            return []

        sample_size = int(self.fraction_evaluate * available)
        sample_size = max(sample_size, self.min_evaluate_clients)
        sample_size = min(sample_size, available)
        min_num = min(self.min_evaluate_clients, sample_size)

        clients = client_manager.sample(sample_size, min_num)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # 转换参数为 numpy arrays 列表（每个客户端的参数列表）
        parameters = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        # 获取每个客户端的样本数量和本地更新次数
        num_examples = [fit_res.num_examples for _, fit_res in results]
        local_steps = [fit_res.metrics.get("local_steps", 1) for _, fit_res in results]

        # 计算归一化系数
        total_examples = sum(num_examples)
        if total_examples == 0:
            return None, {}

        weights = [n / total_examples for n in num_examples]

        # 计算每个客户端的相对步数（tau_eff）
        tau_eff = sum([w * steps for w, steps in zip(weights, local_steps)])

        # FedNova 聚合
        aggregated_ndarrays = []
        num_layers = len(parameters[0])
        for layer in range(num_layers):
            # layer_params: list of numpy arrays for this layer from each client
            layer_params = [p[layer] for p in parameters]

            # 如果该 layer 的 dtype 是整型（比如 int64），直接采用第一个客户端的值（不做浮点聚合）
            first_dtype = layer_params[0].dtype
            if np.issubdtype(first_dtype, np.integer):
                # 直接使用第一个客户端的值（也可以改成取众数或其他策略）
                aggregated = layer_params[0].copy()
                aggregated_ndarrays.append(aggregated)
                continue

            # 否则按浮点执行聚合 —— 初始化浮点累加器
            weighted_sum = np.zeros_like(layer_params[0], dtype=np.float64)

            for i, param in enumerate(layer_params):
                # 强制将客户端参数转换为浮点，避免因某些客户端用 float32/float64 不一致导致问题
                param_f = param.astype(np.float64, copy=False)
                # 归一化客户端更新（tau_eff / local_steps[i]）
                normalized_update = param_f * (tau_eff / max(1, local_steps[i]))
                # 加权累加
                weighted_sum += weights[i] * normalized_update

            # 将浮点累计结果加入聚合列表（保留为 float64）
            aggregated_ndarrays.append(weighted_sum)

        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        # losses 与样本数
        losses = [evaluate_res.loss for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]

        # 加权平均损失（注意避免除以0）
        total_examples = sum(examples)
        if total_examples == 0:
            return None, {}

        aggregated_loss = np.average(losses, weights=examples)

        # 聚合其他指标（如果有）
        aggregated_metrics = {}
        # results 元素为 (client, EvaluateRes)
        if results[0][1].metrics:
            for metric_name in results[0][1].metrics.keys():
                metric_values = [res.metrics[metric_name] for _, res in results]
                aggregated_metrics[metric_name] = np.average(metric_values, weights=examples)

        return aggregated_loss, aggregated_metrics

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)
