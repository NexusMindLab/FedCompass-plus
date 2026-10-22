import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from typing import List

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
    
    # 转换为十分类标签 (0, 1, 2, 3...)
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
    # 默认变换（用于测试集）
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
# data_utils.py (追加内容)

def get_feature_shift_transform(client_id):
    """
    为每个客户端生成特定的特征变换（模拟 Feature Shift）。
    参数设置较为温和，模拟不同设备/环境下的传感器差异。
    """
    # 基础处理：Numpy -> PIL -> [变换] -> Tensor -> 归一化
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # 定义10种不同的变换风格 (对应10个客户端)
    # 强度控制在适中范围，避免模型无法收敛
    augmentations = [
        [],  # Client 0: 原始数据 (Baseline)
        [transforms.ColorJitter(brightness=0.4)],             # Client 1: 亮度变化 (模拟曝光差异)
        [transforms.ColorJitter(contrast=0.4)],               # Client 2: 对比度变化
        [transforms.ColorJitter(saturation=0.4)],             # Client 3: 饱和度变化
        [transforms.RandomRotation(10)],                      # Client 4: 轻微旋转 (+/- 10度)
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], # Client 5: 轻微模糊 (模拟对焦问题)
        [transforms.RandomAffine(degrees=0, translate=(0.0625, 0.0625))], # Client 6: 轻微平移 (模拟构图偏差)
        [transforms.ColorJitter(hue=0.05)],                   # Client 7: 轻微色调漂移 (模拟白平衡差异)
        [transforms.RandomAffine(degrees=0, shear=10)],       # Client 8: 轻微剪切变换 (模拟透视畸变)
        [transforms.RandomPosterize(bits=6, p=1.0)],          # Client 9: 色彩分层 (模拟低位深显示)
    ]
    
    # 根据 client_id 选择对应的变换
    # 如果客户端超过10个，循环使用
    aug_idx = client_id % len(augmentations)
    selected_aug = augmentations[aug_idx]
    
    # 组合变换管道
    transform_list = [to_pil] + selected_aug + [to_tensor, normalize]
    
    return transforms.Compose(transform_list)

def split_data_feature_shift(X, Y, num_clients, batch_size, classes, result_dir):
    """
    实现特征偏移(Feature Shift):
    1. 标签分布：均匀打乱并切分 (IID)
    2. 特征分布：每个客户端应用独特的图像变换 (Non-IID)
    """
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    client_data = {}
    client_class_distributions = {}
    
    # 1. 全局随机打乱数据 (确保标签分布近似 IID)
    # -------------------------------------------------
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # 2. 均匀切分索引给每个客户端
    # -------------------------------------------------
    client_indices_list = np.array_split(indices, num_clients)
    client_sizes = []
    
    # 用于日志描述
    aug_descriptions = [
        "Original", "Brightness", "Contrast", "Saturation", 
        "Rotation(10)", "Blur", "Translate", "Hue", 
        "Shear", "Posterize"
    ]

    print(f"\n[Feature Shift] Starting data split...")

    for client_id in range(num_clients):
        # 获取分片索引
        indices = client_indices_list[client_id]
        client_sizes.append(len(indices))
        
        # 获取数据 (注意：Y 此时已经是 map 过的 0-3 标签，这在 load_data 中完成)
        X_client = X[indices]
        Y_client = Y[indices]
        
        # 3. 关键点：获取该客户端特定的 Transform
        # -------------------------------------------------
        custom_transform = get_feature_shift_transform(client_id)
        
        # 创建数据集
        dataset = CIFAR10Dataset(X_client, Y_client, transform=custom_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_data[client_id] = train_loader
        
        # 记录类别分布 (用于验证标签分布是否均匀)
        counts_full = [0] * len(classes)
        unique, counts = np.unique(Y_client, return_counts=True)
        for u, c in zip(unique, counts):
            counts_full[int(u)] = int(c)
        client_class_distributions[client_id] = counts_full
        
        # 控制台打印进度
        aug_name = aug_descriptions[client_id % len(aug_descriptions)]
        print(f"  Client {client_id} [{aug_name}]: Size {len(indices)}, Label Dist: {counts_full}")

    # 4. 保存分布日志
    # -------------------------------------------------
    with open(f"{result_dir}/client_distribution_feature_shift.txt", "w") as f:
        f.write(f"--- Feature Shift Non-IID Setup ---\n")
        f.write(f"Method: Data shuffled globally (IID Labels) -> Unique transform per client (Non-IID Features)\n")
        f.write(f"Global class distribution: {classes}\n\n")
        
        for client_id in range(num_clients):
            size = client_sizes[client_id]
            class_counts = client_class_distributions[client_id]
            aug_name = aug_descriptions[client_id % len(aug_descriptions)]
            f.write(f"Client {client_id} - Transform: {aug_name} - Size: {size}, Label Dist: {class_counts}\n")
    
    print(f"[Feature Shift] Data splitting complete. Distribution log saved.\n")

    return client_data, client_class_distributions