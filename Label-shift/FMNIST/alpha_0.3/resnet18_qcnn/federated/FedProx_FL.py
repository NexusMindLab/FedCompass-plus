import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
import flwr as fl
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader
from typing import Dict

def client_training(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float,
                    device: torch.device, cid: int, server_round: int, result_dir, 
                    global_params=None, mu=0.0):
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    loss_history = []
    acc_history = []
    
    # 保存全局参数用于FedProx近端项
    if global_params is not None and mu > 0:
        global_weights = {}
        for name, param in zip(model.state_dict().keys(), global_params):
            if "quantum" in name.lower():
                global_weights[name] = torch.tensor(param, dtype=torch.float64, device=device)
            else:
                global_weights[name] = torch.tensor(param, dtype=torch.float32, device=device)
    
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
            
            # 计算交叉熵损失
            loss = criterion(outputs, labels)
            
            # FedProx近端项
            if global_params is not None and mu > 0:
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if name in global_weights:
                        proximal_term += torch.norm(param - global_weights[name]) ** 2
                loss += (mu / 2) * proximal_term
            
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
        algorithm = "FedProx" if mu > 0 else "FedAvg"
        log_message = (
            f"[{current_time}] [{algorithm}] [Client {cid}, Round {server_round}] "
            f"Train Epoch: {epoch+1}/{epochs} \t "
            f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc_percent:.2f}%"
        )
        print(log_message, flush=True)
        
        # 写入文件
        with open(f"{result_dir}/train/client_{cid}_log.txt", "a") as f:
            f.write(
                f"[{current_time}] [{algorithm}] Round {server_round}\t Epoch {epoch + 1}\t "
                f"Loss: {epoch_loss:.4f}\tAccuracy: {epoch_acc_percent:.4f}%\n"
            )
    
    return loss_history, acc_history, model.state_dict()

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

class HybridProxClient(fl.client.NumPyClient):
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
        # 保存全局参数用于FedProx
        global_params = parameters
        self.set_parameters(parameters)
        
        # 从配置中获取参数
        server_round = config.get("server_round", 1)
        local_epochs = config.get("local_epochs", self.config["local_epochs"])
        learning_rate = config.get("learning_rate", self.config["learning_rate"])
        mu = config.get("mu", self.config.get("mu", 0.01))  # FedProx参数
        
        # 客户端训练（带FedProx近端项）
        print(f"[Client {self.cid}] Starting FedProx local training for round {server_round} (mu={mu})")
        loss_hist, acc_hist, model_state = client_training(
            self.model, 
            self.train_loader, 
            local_epochs, 
            learning_rate, 
            self.device,
            self.cid,
            server_round,
            self.config["result_dir"],
            global_params=global_params,
            mu=mu
        )
        
        # 更新本地模型
        self.model.load_state_dict(model_state)
        
        # 返回更新后的参数和结果
        return self.get_parameters({}), len(self.train_loader.dataset), {
            "loss": loss_hist[-1],
            "accuracy": acc_hist[-1]
        }
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = client_evaluation(
            self.model, 
            self.test_loader, 
            self.device
        )
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}

class FedProxHybrid(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, evaluate_fn, mu=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.mu = mu
    
    def initialize_parameters(self, client_manager):
        return self.initial_parameters
    
    def configure_fit(self, server_round, parameters, client_manager):
        """配置客户端训练参数，包含mu值"""
        return super().configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self, server_round, results, failures):
        # 聚合参数（与FedAvg相同）
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters
    
    def evaluate(self, server_round, parameters):
        # 执行服务器端评估
        if self.evaluate_fn is None:
            return None
        
        return self.evaluate_fn(server_round, parameters)