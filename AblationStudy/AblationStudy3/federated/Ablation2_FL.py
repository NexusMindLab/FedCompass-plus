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
# Strategy: Ablation 2 (No Circual Mean -> Standard FedAvg for Quantum Params)
# =============================================================================
class FedAvgAblation2(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters, evaluate_fn, param_keys, num_clusters, qparam_name="q_params",
                 lambda_1: float = 1.0, lambda_2: float = 0.5,   **kwargs):
        super().__init__(**kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.param_keys = param_keys
        self.num_clusters = num_clusters
        self.qparam_name = qparam_name
        self.client_cluster_map = {}
        self.cluster_params = {}
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        # 找到量子参数索引
        init_nds = parameters_to_ndarrays(initial_parameters)
        self.param_shapes = [a.shape for a in init_nds]
        self.q_index = next(i for i, k in enumerate(param_keys) if qparam_name in k)
        self.global_classical = None
        # 注意：这里不需要 global_q 和 FedAdam 状态，因为是算术平均

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return super().aggregate_fit(server_round, results, failures)

        # 解析各客户端返回
        client_ndarrays: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_class_counts: List[np.ndarray] = []
        client_ids: List[str] = []

        for client_proxy, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays.append([np.array(a, dtype=float) for a in nds])
            client_weights.append(fit_res.num_examples)
            metrics = fit_res.metrics or {}
            cc = metrics.get("class_counts", None)
            if cc is None:
                # 若缺失，给一个长度为类别数的均匀向量避免崩溃
                # （你客户端已返回 class_counts，正常不会走到这里）
                num_classes = len(results[0][1].metrics.get("class_counts", [])) or 4
                cc = [1.0] * num_classes
            client_class_counts.append(np.array(cc, dtype=float))

            # ==== 关键修复：正确获取 cid，而不是 client_id ====
            client_ids.append(str(client_proxy.cid))

        num_clients = len(client_ndarrays)
        total_weight = float(sum(client_weights))
        weights = np.array(client_weights, dtype=float) / (total_weight + 1e-12)

        # 1. 准备数据：归一化类别分布
        class_matrix = np.array(client_class_counts, dtype=float)
        row_sums = class_matrix.sum(axis=1, keepdims=True) + 1e-12
        class_matrix_norm = class_matrix / row_sums
        
        # 准备样本量
        sizes = np.array(client_weights, dtype=float)

        # 2. 【核心修改】计算论文公式 (1) 的相似度矩阵
        # S_{ij} = exp( -lambda1 * JS(pi, pj) - lambda2 * |ni - nj|/(ni + nj) )
        S = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    S[i, j] = 1.0
                    continue
                
                # 计算 JS 散度 (scipy 返回的是距离 sqrt(JS)，所以需要平方得到散度值)
                # 如果分布包含 0，jensenshannon 能处理
                js_dist = jensenshannon(class_matrix_norm[i], class_matrix_norm[j], base=2)
                # 防止数值误差
                if np.isnan(js_dist): js_dist = 0.0
                js_div = js_dist ** 2
                
                # 计算样本量差异项
                size_diff = abs(sizes[i] - sizes[j]) / (sizes[i] + sizes[j] + 1e-12)
                
                # 组合公式
                exponent = - self.lambda_1 * js_div - self.lambda_2 * size_diff
                S[i, j] = np.exp(exponent)

        # 3. 执行谱聚类 (affinity='precomputed')
        k = min(self.num_clusters, num_clients)
        if k <= 1:
            labels = np.zeros(num_clients, dtype=int)
        else:
            try:
                sc = SpectralClustering(
                    n_clusters=k, 
                    assign_labels="kmeans", 
                    random_state=42, 
                    affinity="precomputed"  # 【关键修改】使用自定义相似度矩阵
                )
                labels = sc.fit_predict(S)
            except Exception as e:
                print(f"Spectral Clustering failed, fallback to modulo: {e}")
                labels = np.array([i % k for i in range(num_clients)], dtype=int)

        self.client_cluster_map = {client_ids[i]: int(labels[i]) for i in range(num_clients)}
        # 索引划分
        q_index = self.q_index
        classical_indices = [i for i in range(len(self.param_keys)) if i != q_index]

        # ---------- 经典层：计算簇内平均 ----------
        # 准备容器：cluster_id -> 完整参数列表（只填经典层，量子层稍后单独处理）
        self.cluster_params = {c: [None] * len(self.param_keys) for c in range(k)}
        # 同时计算“全局经典平均”（用于评估/首轮下发）
        self.global_classical = [None] * len(self.param_keys)

        # 全局经典平均（标准 FedAvg）
        for idx in classical_indices:
            agg = np.zeros_like(client_ndarrays[0][idx])
            for j in range(num_clients):
                agg += weights[j] * client_ndarrays[j][idx]
            self.global_classical[idx] = agg

        # 簇内平均
        for cluster_id in range(k):
            members = [i for i, lab in enumerate(labels) if lab == cluster_id]
            if len(members) == 0:
                continue
            member_ws = np.array([client_weights[m] for m in members], dtype=float)
            norm_ws = member_ws / (member_ws.sum() + 1e-12)
            for idx in classical_indices:
                agg = np.zeros_like(client_ndarrays[members[0]][idx])
                for a, m in enumerate(members):
                    agg += norm_ws[a] * client_ndarrays[m][idx]
                self.cluster_params[cluster_id][idx] = agg
                    
        # --- 量子参数: 算术平均 (Ablation 2 重点) ---
        # 不使用 Sin/Cos，不使用 FedAdam，直接加权平均
        q_params_list = np.array([client_ndarrays[c][q_index] for c in range(len(results))])
        # global_q_arithmetic = sum(w_i * q_i)
        global_q_arithmetic = np.sum(q_params_list * weights[:, None], axis=0)
        
        # 组装结果
        final_params = []
        for i in range(len(self.param_keys)):
            if i == q_index:
                final_params.append(global_q_arithmetic)
            else:
                final_params.append(self.global_classical[i])
                
        # 更新 cluster_params 里的量子参数为全局算术平均 (因为我们只做经典聚类)
        # 注意：configure_fit 逻辑需要适配，因为 cluster_params 原本是存簇级经典的
        # 这里的 cluster_params 只存了经典层，configure_fit 会取全局量子层
        
        return ndarrays_to_parameters(final_params), {}

    def configure_fit(self, server_round, parameters, client_manager):
        # 复用 Ours 的 configure_fit 逻辑
        # 核心区别在于 parameters 里的量子参数是算术平均得来的
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)
        if not self.client_cluster_map: return fit_instructions
        
        q_index = self.q_index
        global_params = parameters_to_ndarrays(parameters) # 包含算术平均的量子参数
        
        for client_proxy, fit_ins in fit_instructions:
            cid = str(client_proxy.cid)
            cluster_id = self.client_cluster_map.get(cid, None)
            
            new_params = []
            for i in range(len(self.param_keys)):
                if i == q_index:
                    new_params.append(global_params[i]) # 下发算术平均的量子参数
                else:
                    if cluster_id is not None and self.cluster_params[cluster_id][i] is not None:
                        new_params.append(self.cluster_params[cluster_id][i])
                    else:
                        new_params.append(global_params[i])
            fit_ins.parameters = ndarrays_to_parameters(new_params)
            
        return fit_instructions

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)