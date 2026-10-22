import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from typing import List

class FashionMNISTDataset(Dataset):  # 修改类名
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data(classes):
    # 修改为加载FashionMNIST数据集
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.fashion_mnist.load_data()
    
    # FashionMNIST数据需要添加通道维度
    x_train_all = np.expand_dims(x_train_all, axis=-1)
    x_test_all = np.expand_dims(x_test_all, axis=-1)
    
    # 筛选指定类别
    train_mask = np.isin(y_train_all, classes)
    test_mask = np.isin(y_test_all, classes)
    
    x_train = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    x_test = x_test_all[test_mask]
    y_test = y_test_all[test_mask]
    
    # 转换为十分类标签
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    y_train = np.vectorize(class_mapping.get)(y_train)
    y_test = np.vectorize(class_mapping.get)(y_test)
    
    print(f"Original train set: {x_train_all.shape[0]}, Filtered: {len(x_train)}")
    print(f"Original test set: {x_test_all.shape[0]}, Filtered: {len(x_test)}")
    print(f"Class distribution - Train: {np.unique(y_train, return_counts=True)}")
    print(f"Class distribution - Test: {np.unique(y_test, return_counts=True)}")
    
    # # 测试代码
    # x_train = x_train[:200]
    # y_train = y_train[:200]
    # x_test = x_test[:min(200, len(x_test))]
    # y_test = y_test[:min(200, len(y_test))]
    
    return x_train, y_train, x_test, y_test

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 修改为单通道的归一化
    ])

def split_data_to_clients(X, Y, num_clients, batch_size, classes, alpha, result_dir):
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 创建每个客户端的数据集
    client_data = {}
    client_class_distributions = {}
    transform = get_transform()

    # 获取类别标签
    y_labels = np.array(Y).flatten()
    unique_classes = np.unique(y_labels).tolist()
    
    # 初始化客户端数据字典
    client_indices = {i: [] for i in range(num_clients)}

    # 对每个类别使用Dirichlet分布分配样本
    for cls in unique_classes:
        cls_indices = np.where(y_labels == cls)[0]
        n_cls_samples = len(cls_indices)

        if n_cls_samples == 0:
            continue

        # 使用Dirichlet分布生成客户端分配比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # 计算每个客户端分配的数量
        allocations = np.cumsum(proportions * n_cls_samples).astype(int)
        allocations = np.clip(allocations, 0, n_cls_samples)

        # 将样本分配给客户端
        start = 0
        for client_id in range(num_clients):
            end = allocations[client_id]
            if end > start:
                client_indices[client_id].extend(cls_indices[start:end])
            start = end

    # 客户端类别分布
    class_distributions = {}
    client_sizes = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_sizes.append(len(indices))

        # 获取客户端数据
        X_client = X[indices]
        y_client = y_labels[indices]

        # 保持 classes 映射一致
        class_mapping = {cls: idx for idx, cls in enumerate(classes)}
        y_client = np.vectorize(class_mapping.get)(y_client)
        
        # 创建数据集
        dataset = FashionMNISTDataset(X_client, y_client, transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_data[client_id] = train_loader

        # 记录类别分布
        class_counts = {int(c): int(np.sum(y_labels[indices] == c)) for c in unique_classes}
        class_distributions[client_id] = class_counts
        # 为了确保对齐 classes 长度，构造长度为 len(classes)
        counts_full = [0] * len(classes)
        for idx, cls in enumerate(classes):
            counts_full[idx] = int(np.sum(y_client == idx))
        client_class_distributions[client_id] = counts_full

    # 保存分布信息
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