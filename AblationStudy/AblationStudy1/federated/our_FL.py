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
import os

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
    # 【新增】记录量子梯度方差历史
    q_grad_var_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        # 【新增】记录当前 epoch 内每个 batch 的量子梯度方差
        batch_q_vars = []
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

            # ================== 【新增】计算量子梯度方差 (在动态平衡之前测量) ==================
            # 这里的目的是记录“当前参数位置下的梯度性质”，反映训练的稳定性
            if model.q_params.grad is not None:
                # view(-1) 拉平后计算方差
                q_var = torch.var(model.q_params.grad.view(-1)).item()
                batch_q_vars.append(q_var)
            else:
                batch_q_vars.append(0.0)
            # ==============================================================================

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
        # 【新增】计算本 epoch 平均方差
        avg_q_var = sum(batch_q_vars) / len(batch_q_vars) if batch_q_vars else 0.0
        q_grad_var_history.append(avg_q_var)
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
            
    # 【修改】返回 q_grad_var_history
    return loss_history, acc_history, q_grad_var_history, model.state_dict()

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
        # 【修改】接收 q_var_hist
        loss_hist, acc_hist, q_var_hist, model_state = client_training(
            self.model, 
            self.train_loader, 
            local_epochs, 
            learning_rate, 
            self.device,
            self.cid,
            server_round,
            self.config["result_dir"]
        )
        self.model.load_state_dict(model_state)
        
        # 【新增】计算本轮平均梯度方差
        avg_q_var_round = sum(q_var_hist) / len(q_var_hist) if q_var_hist else 0.0

        metrics = {
            "loss": loss_hist[-1] if len(loss_hist)>0 else None,
            "accuracy": acc_hist[-1] if len(acc_hist)>0 else None,
            "class_counts": np.array(self.class_counts, dtype=int).tolist(),
            "q_grad_variance": float(avg_q_var_round) # 【新增】返回 Metrics
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

# ================== 自定义策略：簇聚合 + 量子圆均值 + FedAdam + 按簇下发 ==================
class FedAvgHybridClustered(fl.server.strategy.FedAvg):
    def __init__(
        self,
        initial_parameters,
        evaluate_fn,
        param_keys: List[str],
        num_clusters: int = 3,
        qparam_name: str = "q_params",
        lr_q: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        lambda_1: float = 1.0,  # 【新增】JS散度权重
        lambda_2: float = 0.5,  # 【新增】样本量差异权重
        result_dir: str = "result/Ours",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.param_keys = list(param_keys)
        self.num_clusters = num_clusters
        self.qparam_name = qparam_name
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.result_dir = result_dir
        # FedAdam states for q_params
        self.lr_q = lr_q
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None

        # 形状与索引
        init_ndarrays = parameters_to_ndarrays(self.initial_parameters)
        self.param_shapes = [np.array(a).shape for a in init_ndarrays]
        try:
            self.q_index = self.param_keys.index(self.qparam_name)
        except ValueError:
            self.q_index = next(
                (i for i, k in enumerate(self.param_keys) if "q_param" in k or "qparams" in k or "q_params" in k),
                None
            )
        if self.q_index is None:
            raise ValueError("Could not find quantum parameter key in param_keys.")
        self.global_q = np.array(init_ndarrays[self.q_index], dtype=float)

        # 用于“按簇下发”的持久化状态（在 configure_fit 使用）
        self.client_cluster_map: Dict[str, int] = {}      # cid(str) -> cluster_id
        self.cluster_params: Dict[int, List[np.ndarray]] = {}  # cluster_id -> full param list（经典层）
        self.global_classical: List[np.ndarray] = None     # 经典层的全局平均（用于评估/首轮下发）

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

        # 【新增】聚合梯度方差
        total_q_var = 0.0
        total_examples_for_var = 0

        for client_proxy, fit_res in results:
            nds = parameters_to_ndarrays(fit_res.parameters)
            client_ndarrays.append([np.array(a, dtype=float) for a in nds])
            client_weights.append(fit_res.num_examples)
            metrics = fit_res.metrics or {}
            cc = metrics.get("class_counts", None)
            # 【新增】读取 Q_Var
            q_var = metrics.get("q_grad_variance", 0.0)
            total_q_var += q_var * fit_res.num_examples
            total_examples_for_var += fit_res.num_examples

            if cc is None:
                # 若缺失，给一个长度为类别数的均匀向量避免崩溃
                # （你客户端已返回 class_counts，正常不会走到这里）
                num_classes = len(results[0][1].metrics.get("class_counts", [])) or 4
                cc = [1.0] * num_classes
            client_class_counts.append(np.array(cc, dtype=float))

            # ==== 关键修复：正确获取 cid，而不是 client_id ====
            client_ids.append(str(client_proxy.cid))

        # 【新增】写入聚合后的方差到文件
        if total_examples_for_var > 0:
            avg_global_q_var = total_q_var / total_examples_for_var
            os.makedirs(f"{self.result_dir}/test", exist_ok=True)
            with open(f"{self.result_dir}/test/server_q_variance.txt", "a") as f:
                f.write(f"Round {server_round}\tAverage_Q_Variance: {avg_global_q_var:.8f}\n")
            print(f"Round {server_round} aggregated Quantum Gradient Variance: {avg_global_q_var:.8f}")
            
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

        # ---------- 量子层：圆均值 + FedAdam ----------
        q_vectors = np.array([client_ndarrays[i][q_index].reshape(-1) for i in range(num_clients)])
        sin_sum = np.sum((np.sin(q_vectors) * np.array(client_weights)[:, None]), axis=0)
        cos_sum = np.sum((np.cos(q_vectors) * np.array(client_weights)[:, None]), axis=0)
        q_circular = np.arctan2(sin_sum, cos_sum)
        q_circular = (q_circular + np.pi) % (2 * np.pi) - np.pi  # wrap 到 (-π, π]

        grad = (q_circular - self.global_q)
        grad = (grad + np.pi) % (2 * np.pi) - np.pi

        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** server_round)
        v_hat = self.v / (1 - self.beta2 ** server_round)
        update = self.lr_q * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.global_q = self.global_q - update
        self.global_q = (self.global_q + np.pi) % (2 * np.pi) - np.pi

        # 组装“服务器持有的全局参数”：经典=全局平均；量子=全局量子
        aggregated_ndarrays = []
        for i in range(len(self.param_keys)):
            if i == q_index:
                aggregated_ndarrays.append(np.array(self.global_q, dtype=float).reshape(self.param_shapes[i]))
            else:
                aggregated_ndarrays.append(np.array(self.global_classical[i], dtype=float))

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # 注意：返回值必须是 (Parameters, metrics)
        # 我们把聚类信息记录到 self.client_cluster_map/self.cluster_params，
        # 下一轮在 configure_fit 里按簇下发
        metrics = {"num_clients": num_clients, "num_clusters": int(k)}
        return aggregated_parameters, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        """按簇向每个被抽样的客户端下发其簇的经典参数 + 全局量子参数。"""
        # 先用父类完成采样与默认 FitIns
        fit_instructions = super().configure_fit(server_round, parameters, client_manager)

        # 首轮还没有聚类信息，保持默认
        if not self.client_cluster_map or not self.cluster_params:
            return fit_instructions

        # 构造每个客户端个性化下发参数
        q_index = self.q_index
        for i, (client_proxy, fit_ins) in enumerate(fit_instructions):
            cid = str(client_proxy.cid)
            cluster_id = self.client_cluster_map.get(cid, None)

            # 若没命中（极少数情况），回退到全局经典
            if cluster_id is None or cluster_id not in self.cluster_params:
                per_params = []
                for idx in range(len(self.param_keys)):
                    if idx == q_index:
                        per_params.append(np.array(self.global_q, dtype=float).reshape(self.param_shapes[idx]))
                    else:
                        per_params.append(np.array(self.global_classical[idx], dtype=float))
            else:
                # 该簇的经典 + 全局量子
                per_params = []
                for idx in range(len(self.param_keys)):
                    if idx == q_index:
                        per_params.append(np.array(self.global_q, dtype=float).reshape(self.param_shapes[idx]))
                    else:
                        # 某些 buffer（如 BN 统计）在极端情况下可能 None，回退到全局经典
                        val = self.cluster_params[cluster_id][idx]
                        if val is None:
                            val = self.global_classical[idx]
                        per_params.append(np.array(val, dtype=float))

            # 替换成个性化下发参数
            fit_ins.parameters = ndarrays_to_parameters(per_params)
            # 可选：把 cluster_id 写进 config 方便客户端侧日志
            fit_ins.config = dict(fit_ins.config or {})
            fit_ins.config["cluster_id"] = int(cluster_id) if cluster_id is not None else -1

        return fit_instructions

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)