import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import os
from typing import List

class TabularDataset(Dataset):
    """表格数据集包装器"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_transform():
    """表格数据不需要图像变换"""
    return None

def load_data(classes=None):
    """加载 Breast Cancer Wisconsin 数据集。

    Args:
        classes: 忽略 (兼容接口), Breast Cancer 固定为 2 类
    Returns:
        x_train, y_train, x_test, y_test
    """
    data = load_breast_cancer()
    X, y = data.data, data.target  # (569, 30), (569,)

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 80/20 分割
    n = len(X)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    x_train, y_train = X[train_idx], y[train_idx]
    x_test, y_test = X[test_idx], y[test_idx]

    print(f"Breast Cancer Wisconsin:")
    print(f"  Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"  Features: {x_train.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"  Class dist Train: {np.bincount(y_train)}")
    print(f"  Class dist Test:  {np.bincount(y_test)}")

    return x_train, y_train, x_test, y_test

def split_data_to_clients(X, Y, num_clients, batch_size, classes, alpha, result_dir):
    """Dirichlet 分布分配客户端数据 (Label-shift 模式)"""
    os.makedirs(result_dir, exist_ok=True)

    client_data = {}
    client_class_distributions = {}
    y_labels = np.array(Y).flatten()
    unique_classes = np.unique(y_labels).tolist()

    client_indices = {i: [] for i in range(num_clients)}

    for cls in unique_classes:
        cls_indices = np.where(y_labels == cls)[0]
        n_cls_samples = len(cls_indices)
        if n_cls_samples == 0:
            continue

        proportions = np.random.RandomState(42).dirichlet(np.repeat(alpha, num_clients))
        allocations = np.cumsum(proportions * n_cls_samples).astype(int)
        allocations = np.clip(allocations, 0, n_cls_samples)

        start = 0
        for client_id in range(num_clients):
            end = allocations[client_id]
            if end > start:
                client_indices[client_id].extend(cls_indices[start:end])
            start = end

    # 修复：Dirichlet 比例截断可能导致部分客户端样本过少甚至为 0。
    # 空客户端会让 vectorize/DataLoader 崩溃；样本数=1 时 batch_size=1，
    # BatchNorm 每轮都会崩溃（实测 3 个 1 样本客户端每轮全部失败，只收到 7/10 结果）。
    # 从样本最多的客户端向样本不足 2 个的客户端匀样本，保证每个客户端至少 2 个样本。
    for client_id in range(num_clients):
        while len(client_indices[client_id]) < 2:
            donor = max(range(num_clients), key=lambda i: len(client_indices[i]))
            if len(client_indices[donor]) <= 2:
                raise ValueError(
                    f"总样本数不足以让 {num_clients} 个客户端各持有至少 2 个样本")
            client_indices[client_id].append(client_indices[donor].pop())

    class_distributions = {}
    client_sizes = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_sizes.append(len(indices))
        X_client = X[indices]
        y_client = y_labels[indices]

        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        # 修复：np.vectorize 在空数组上会报错，改为列表推导（对空数组也安全）
        y_client = np.array([class_mapping[y] for y in y_client], dtype=np.int64)

        dataset = TabularDataset(X_client, y_client)
        train_loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        client_data[client_id] = train_loader

        class_counts = {int(c): int(np.sum(y_labels[indices] == c)) for c in unique_classes}
        class_distributions[client_id] = class_counts
        counts_full = [0] * len(classes)
        for idx, cls in enumerate(classes):
            counts_full[idx] = int(np.sum(y_client == idx))
        client_class_distributions[client_id] = counts_full

    with open(f"{result_dir}/client_distribution.txt", "w") as f:
        f.write(f"Global class distribution: {unique_classes}\n")
        f.write(f"Client data sizes: {client_sizes}\n\n")
        for client_id in range(num_clients):
            size = client_sizes[client_id]
            class_counts = class_distributions[client_id]
            f.write(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}\n")

    print(f"\nGlobal class distribution: {unique_classes}")
    print(f"Client data sizes: {client_sizes}")
    for client_id in range(num_clients):
        size = client_sizes[client_id]
        class_counts = class_distributions[client_id]
        print(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}")

    return client_data, client_class_distributions
