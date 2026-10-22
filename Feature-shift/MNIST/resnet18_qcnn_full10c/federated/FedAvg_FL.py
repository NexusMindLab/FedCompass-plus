import os
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
                    device: torch.device, cid: int, server_round: int, result_dir, local_log_dir=None):
    model.to(device)
    model.train()

    # 使用本地临时目录写日志，避免NFS并发I/O错误
    _log_dir = local_log_dir if local_log_dir else result_dir
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    loss_history = []
    acc_history = []
    
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
        # 写入文件（使用本地临时目录避免NFS并发问题）
        os.makedirs(f"{_log_dir}/train", exist_ok=True)
        with open(f"{_log_dir}/train/client_{cid}_log.txt", "a") as f:
            f.write(
                f"[{current_time}] Round {server_round}\t Epoch {epoch + 1}\t "
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
        loss_hist, acc_hist, model_state = client_training(
            self.model, 
            self.train_loader, 
            local_epochs, 
            learning_rate, 
            self.device,
            self.cid,
            server_round,
            self.config["result_dir"],
            self.config.get("local_log_dir")
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

class FedAvgHybrid(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, evaluate_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
    
    def initialize_parameters(self, client_manager):
        return self.initial_parameters
    
    def aggregate_fit(self, server_round, results, failures):
        # 聚合参数
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        return aggregated_parameters
    
    def evaluate(self, server_round, parameters):
        # 执行服务器端评估
        if self.evaluate_fn is None:
            return None
        
        return self.evaluate_fn(server_round, parameters)