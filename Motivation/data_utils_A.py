import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from torchvision import transforms
import os
from typing import List
import random


class CIFAR10Dataset(Dataset):
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

def load_data(alpha=None, classes=None, transform=None):
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar10.load_data()
    y_train_all, y_test_all = y_train_all.flatten(), y_test_all.flatten()
    
    # 筛选指定类别
    train_mask = np.isin(y_train_all, classes)
    test_mask = np.isin(y_test_all, classes)
    
    x_train = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    x_test = x_test_all[test_mask]
    y_test = y_test_all[test_mask]
    
    # 转换为四分类标签
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
    
    # 如果提供了transform，创建Dataset对象
    if transform is not None:
        train_dataset = CIFAR10Dataset(x_train, y_train, transform=transform)
        test_dataset = CIFAR10Dataset(x_test, y_test, transform=transform)
        return train_dataset, test_dataset
    else:
        return (x_train, y_train, x_test, y_test)

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def split_data_to_clients(train_dataset, num_clients, result_dir, iid=True):
    """分割数据到客户端，兼容exp_A.py的调用方式"""
    # 获取原始数据
    X = train_dataset.images
    Y = train_dataset.labels
    transform = train_dataset.transform
    
    if iid:
        # IID分割
        return split_data_to_clients_IID(X, Y, num_clients, list(np.unique(Y)), result_dir)
    else:
        # 非IID分割，使用默认alpha=0.3
        return split_data_to_clients_nonIID(X, Y, num_clients, list(np.unique(Y)), 0.3, result_dir)

def split_data_to_clients_nonIID(X, Y, num_clients, classes, alpha, result_dir):
    """非IID数据分割"""
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    # 获取类别标签
    y_labels = np.array(Y).flatten()
    unique_classes = np.unique(y_labels).tolist()
    
    # 初始化客户端数据字典
    client_indices = {i: [] for i in range(num_clients)}
    
    # 对每个类别使用Dirichlet分布分配样本
    for cls in unique_classes:
        # 获取当前类别的所有样本索引
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
    
    # 创建每个客户端的数据集
    client_class_distributions = {}
    client_sizes = []
    client_datasets = {}
    
    transform = get_transform()
    
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        client_sizes.append(len(indices))
        
        # 获取客户端数据
        X_client = X[indices]
        y_client = y_labels[indices]
        
        # 创建数据集
        dataset = CIFAR10Dataset(X_client, y_client, transform=transform)
        client_datasets[client_id] = dataset
        
        # 记录类别分布
        class_counts = {int(c): int(np.sum(y_labels[indices] == c)) for c in unique_classes}
        client_class_distributions[client_id] = class_counts
    
    # 保存分布信息
    with open(f"{result_dir}/client_distribution.txt", "w") as f:
        f.write(f"Global class distribution: {unique_classes}\n")
        f.write(f"Client data sizes: {client_sizes}\n\n")
        
        for client_id in range(num_clients):
            size = client_sizes[client_id]
            class_counts = client_class_distributions[client_id]
            f.write(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}\n")
    
    print(f"\nGlobal class distribution: {unique_classes}")
    print(f"Client data sizes: {client_sizes}")
    
    for client_id in range(num_clients):
        size = client_sizes[client_id]
        class_counts = client_class_distributions[client_id]
        print(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}")
    
    return client_datasets

def split_data_to_clients_IID(X, Y, num_clients, classes, result_dir):
    """IID数据分割"""
    client_datasets = {}
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    transform = get_transform()
    
    client_sizes = [len(X) // num_clients for _ in range(num_clients)]
    for i in range(len(X) % num_clients):
        client_sizes[i] += 1
        
    y_labels = np.array(Y).flatten()
    unique_classes = np.unique(y_labels).tolist()
        
    start_idx = 0
    client_class_distributions = {}
    for client_id in range(num_clients):
        end_idx = start_idx + client_sizes[client_id]
        client_indices = indices[start_idx:end_idx]
        X_client = X[client_indices]
        y_client = y_labels[client_indices]
            
        # 创建数据集
        dataset = CIFAR10Dataset(X_client, y_client, transform=transform)
        client_datasets[client_id] = dataset
        
        class_counts = {int(cls): int(np.sum(y_client == cls)) for cls in unique_classes}
        client_class_distributions[client_id] = class_counts
        start_idx = end_idx
    
    with open(f"{result_dir}/client_distribution.txt", "a") as f:
        f.write(f"Global class distribution: {unique_classes}\n")
        f.write(f"Client data sizes: {client_sizes}\n\n")
        
        for client_id in range(num_clients):
            size = client_sizes[client_id]
            class_counts = client_class_distributions[client_id]
            f.write(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}\n")

    print(f"Global class distribution: {unique_classes}")
    print(f"Client data sizes: {client_sizes}")
        
    for client_id in range(num_clients):
        size = client_sizes[client_id]
        class_counts = client_class_distributions[client_id]
        print(f"Client {client_id} - Size: {size}, Class distribution: {class_counts}")

    return client_datasets