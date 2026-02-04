import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from sklearn.cluster import SpectralClustering
from typing import Dict, List
from torch.utils.data import DataLoader
from scipy.spatial.distance import jensenshannon

def client_training(model: nn.Module, train_loader: DataLoader, epochs: int, learning_rate: float,
                    device: torch.device, cid: int, server_round: int, result_dir,):
    model.to(device)
    model.train()
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # ================== 改进点 1: 分层学习率 (Layer-wise Learning Rates) ==================
    # 区分经典参数和量子参数
    # 注意：model是HybridModelX实例，包含 feature_extractor 和 q_params
    classical_params = list(model.feature_extractor.parameters())
    # q_params 是 nn.Parameter，需要放入列表
    quantum_params = [model.q_params]

    # 设定量子学习率倍率 (建议 10-30 倍)
    lr_mult = 10.0 
    
    optimizer = optim.Adam([
        {'params': classical_params, 'lr': learning_rate},        # 经典层保持原 LR (0.001)
        {'params': quantum_params, 'lr': learning_rate * lr_mult} # 量子层放大 LR (0.01)
    ])
    # ===================================================================================

    # 学习率调度器 (针对 optimizer 中的所有 param_group 进行调整)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
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

            # ================== 改进点 3: 动态梯度平衡 (Dynamic Gradient Balancing) ==================
            # 在梯度裁剪和更新之前执行
            
            # 1. 计算经典层梯度范数 (L2 Norm)
            # 过滤掉 None 的梯度
            c_grads = [p.grad for p in classical_params if p.grad is not None]
            if c_grads:
                c_norm = torch.norm(torch.stack([torch.norm(g) for g in c_grads]))
            else:
                c_norm = torch.tensor(0.0, device=device)

            # 2. 计算量子层梯度范数
            if model.q_params.grad is not None:
                q_norm = torch.norm(model.q_params.grad)
            else:
                q_norm = torch.tensor(0.0, device=device)

            # 3. 诊断与缩放触发逻辑
            threshold_ratio = 10.0 
            
            # 确保分母不为0且存在梯度
            if q_norm > 1e-9 and c_norm > 1e-9:
                current_imbalance = c_norm / q_norm
                
                # 如果失衡超过阈值 (例如 > 10x)
                if current_imbalance > threshold_ratio:
                    # A. 计算缩放因子: 让放大后的 Q_norm 达到 C_norm / threshold
                    scale_factor = current_imbalance / threshold_ratio
                    
                    # B. 安全截断 (保险丝) - 防止缩放过大导致发散
                    scale_factor = torch.clamp(scale_factor, max=100.0)
                    
                    # C. 执行缩放 (In-place)
                    model.q_params.grad.data.mul_(scale_factor)
            # =======================================================================================

            # ================== 改进点 2: 差异化梯度裁剪 (Differential Gradient Clipping) ==================
            # 仅对经典层进行严格裁剪，防止其主导更新
            torch.nn.utils.clip_grad_norm_(classical_params, max_norm=1.0)
            
            # 量子层跳过裁剪 (因为量子梯度通常很小，且已经过上面的动态平衡处理)
            # ==========================================================================================

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
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"[{current_time}] [Client {cid}, Round {server_round}] "
            f"Train Epoch: {epoch+1}/{epochs} \t Loss: {epoch_loss:.4f} | Acc: {epoch_acc_percent:.2f}%"
        )
        print(log_message, flush=True)
        with open(f"{result_dir}/train/client_{cid}_log.txt", "a") as f:
            f.write(f"[{current_time}] Round {server_round}\t Epoch {epoch+1}\t Loss: {epoch_loss:.4f}\tAccuracy: {epoch_acc_percent:.4f}%\n")
            
    return loss_history, acc_history, model.state_dict()

def client_evaluation(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0  # 更明确的命名
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()  # ✅ 直接转为long，避免冗余
            
            outputs = model(images)
            
            # 计算损失
            total_loss += criterion(outputs, labels).item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    if len(test_loader) == 0:
        return 0.0, 0.0  # 空数据集的默认值
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

class HybridClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, model, device, config, class_counts):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.config = config
        self.class_counts = class_counts  # 本地类别计数（列表）
        
    def get_parameters(self, config):
        # 返回模型参数
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        state_dict = {}
        for k, v in zip(self.model.state_dict().keys(), parameters):
            state_dict[k] = torch.tensor(v, dtype=self.model.state_dict()[k].dtype)
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        # 设置参数
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
            self.config["result_dir"]
        )
        # 更新本地模型
        self.model.load_state_dict(model_state)
        # 返回参数，同时在 metrics 中返回 class distribution，供服务器聚类使用
        metrics = {
            "loss": loss_hist[-1] if len(loss_hist)>0 else None,
            "accuracy": acc_hist[-1] if len(acc_hist)>0 else None,
            "class_counts": np.array(self.class_counts, dtype=int).tolist()
        }
        return self.get_parameters({}), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # 客户端评估
        loss, accuracy = client_evaluation(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

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
    total_loss = 0.0  # 更清晰的命名
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()  # ✅ 直接转换为long
            
            outputs = model(images)
            
            # 计算损失
            total_loss += criterion(outputs, labels).item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    if len(test_loader) == 0:
        return 0.0, {"accuracy": 0.0}
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nServer round {server_round}: Evaluation loss {avg_loss:.4f} / accuracy {accuracy*100:.4f} %")
    
    with open(f"{result_dir}/test/server_test.txt", "a") as f:
        f.write(f"Round {server_round}\tLoss: {avg_loss:.4f}\tAccuracy: {accuracy*100:.4f}%\n")
    
    return avg_loss, {"accuracy": accuracy}

# =============================================================================
# Strategy: Ablation 1 (No Clustering -> Standard FedAvg for Classical)
# =============================================================================
class FedAvgAblation1(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, evaluate_fn, param_keys, qparam_name="q_params", lr_q=0.001, **kwargs):
        super().__init__(**kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.param_keys = param_keys
        
        # FedAdam states for quantum
        self.lr_q = lr_q
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.m, self.v = None, None
        
        # Init quantum global
        init_nds = parameters_to_ndarrays(initial_parameters)
        self.param_shapes = [a.shape for a in init_nds]
        # 寻找量子参数索引
        try:
            self.q_index = next(i for i, k in enumerate(param_keys) if qparam_name in k)
        except StopIteration:
            raise ValueError(f"Quantum parameter '{qparam_name}' not found in param_keys.")
            
        self.global_q = np.array(init_nds[self.q_index], dtype=float)

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        if not results: return super().aggregate_fit(server_round, results, failures)
        
        # 1. 收集参数
        client_ndarrays = [parameters_to_ndarrays(res.parameters) for _, res in results]
        weights = np.array([res.num_examples for _, res in results], dtype=float)
        # 归一化权重
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)
        
        aggregated_ndarrays = []
        
        # 2. 遍历每一层参数
        for i, key in enumerate(self.param_keys):
            if i == self.q_index:
                # === 量子层: Circular Mean + FedAdam (保留) ===
                q_vectors = np.array([client_ndarrays[c][i].reshape(-1) for c in range(len(results))])
                
                # 广播 weights 用于 sin/cos 加权求和
                sin_sum = np.sum(np.sin(q_vectors) * weights[:, None], axis=0)
                cos_sum = np.sum(np.cos(q_vectors) * weights[:, None], axis=0)
                q_circular = np.arctan2(sin_sum, cos_sum)
                
                # FedAdam Update
                grad = (q_circular - self.global_q + np.pi) % (2*np.pi) - np.pi
                if self.m is None: self.m, self.v = np.zeros_like(grad), np.zeros_like(grad)
                
                self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
                m_hat = self.m / (1 - self.beta1 ** server_round)
                v_hat = self.v / (1 - self.beta2 ** server_round)
                
                self.global_q = self.global_q + self.lr_q * m_hat / (np.sqrt(v_hat) + self.epsilon)
                self.global_q = (self.global_q + np.pi) % (2*np.pi) - np.pi
                
                aggregated_ndarrays.append(self.global_q.reshape(self.param_shapes[i]))
            else:
                # === 经典层: Standard FedAvg (全局平均，无聚类) ===
                layer_params = np.array([client_ndarrays[c][i] for c in range(len(results))])
                
                # 【修正点】通用广播逻辑：构造形状 (num_clients, 1, 1, ..., 1)
                # 例如 layer_params 是 (10, 64, 3, 3, 3) -> 5D
                # weights 是 (10,) -> 变为 (10, 1, 1, 1, 1)
                broadcast_shape = (len(weights),) + (1,) * (layer_params.ndim - 1)
                w_reshaped = weights.reshape(broadcast_shape)
                
                avg_param = np.sum(layer_params * w_reshaped, axis=0)
                aggregated_ndarrays.append(avg_param)
                
        return ndarrays_to_parameters(aggregated_ndarrays), {}

    def evaluate(self, server_round, parameters):
        return self.evaluate_fn(server_round, parameters) if self.evaluate_fn else None