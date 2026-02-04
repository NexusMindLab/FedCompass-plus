# data_utils.py
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

def load_data(classes):
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar10.load_data()
    y_train_all, y_test_all = y_train_all.flatten(), y_test_all.flatten()
    
    # 筛选指定类别
    train_mask = np.isin(y_train_all, classes)
    test_mask = np.isin(y_test_all, classes)
    
    x_train = x_train_all[train_mask]
    y_train = y_train_all[train_mask]
    x_test = x_test_all[test_mask]
    y_test = y_test_all[test_mask]
    
    # 转换为标签索引 (0, 1, 2...)
    class_mapping = {cls: idx for idx, cls in enumerate(classes)}
    y_train = np.vectorize(class_mapping.get)(y_train)
    y_test = np.vectorize(class_mapping.get)(y_test)
    
    print(f"Original train set: {x_train_all.shape[0]}, Filtered: {len(x_train)}")
    print(f"Original test set: {x_test_all.shape[0]}, Filtered: {len(x_test)}")
    
    # # 测试代码
    # x_train = x_train[:200]
    # y_train = y_train[:200]
    # x_test = x_test[:min(200, len(x_test))]
    # y_test = y_test[:min(200, len(y_test))]

    return x_train, y_train, x_test, y_test

def get_transform():
    """定义统一的CIFAR-10预处理"""
    return transforms.Compose([
        transforms.ToTensor(),
        # 修正：CIFAR-10是RGB三通道，对应三个均值和标准差
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def split_data_to_clients_IID(X, Y, num_clients, classes, result_dir):
    """
    IID数据分割逻辑
    修复点：使用统一的 get_transform()，避免硬编码错误的单通道归一化
    """
    client_datasets = {} 
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # 【重要修复】：使用统一的三通道 transform
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
    
    # 保存分布信息
    os.makedirs(result_dir, exist_ok=True)
    with open(f"{result_dir}/client_distribution.txt", "w") as f:
        f.write(f"Global class distribution: {unique_classes}\n")
        f.write(f"Client data sizes: {client_sizes}\n")
        for client_id in range(num_clients):
            f.write(f"Client {client_id}: {client_class_distributions[client_id]}\n")

    print(f"Client data sizes: {client_sizes}")
    return client_datasets