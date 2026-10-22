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

def get_feature_shift_transform(client_id):
    """
    针对 Fashion-MNIST (灰度图) 的特征偏移变换
    """
    # 1. 基础转换
    # 输入是 (28, 28, 1) 的 numpy 数组，ToPILImage 会自动将其转为 Mode 'L' (灰度)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    # 【修改点1】归一化改为单通道
    normalize = transforms.Normalize((0.5,), (0.5,)) 
    
    # 2. 定义10种适合灰度图的变换风格
    augmentations = [
        [],  # Client 0: 原始数据
        
        # 亮度与对比度 (灰度图依然有效)
        [transforms.ColorJitter(brightness=0.5)],             # Client 1: 亮度变化
        [transforms.ColorJitter(contrast=0.5)],               # Client 2: 对比度变化
        
        # 几何变换
        [transforms.RandomRotation(15)],                      # Client 3: 旋转 (+/- 15度)
        [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], # Client 4: 平移
        [transforms.RandomAffine(degrees=0, shear=15)],       # Client 5: 剪切变换
        
        # 模糊与噪声
        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], # Client 6: 高斯模糊
        
        # 【修改点2】新增适合灰度图的变换 (替代 Hue/Saturation)
        [transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0)], # Client 7: 锐化
        [transforms.RandomPosterize(bits=4, p=1.0)],          # Client 8: 色调分离 (减少灰度级)
        
        # 极端变换 (Fashion-MNIST 结构简单，反色通常也能识别，模拟黑白底色反转)
        [transforms.RandomInvert(p=1.0)],                     # Client 9: 反色
    ]
    
    # 根据 client_id 选择对应的变换
    aug_idx = client_id % len(augmentations)
    selected_aug = augmentations[aug_idx]
    
    # 组合变换管道
    transform_list = [to_pil] + selected_aug + [to_tensor, normalize]
    
    return transforms.Compose(transform_list)

def split_data_feature_shift(X, Y, num_clients, batch_size, classes, result_dir):
    """
    Fashion-MNIST 特征偏移划分：
    标签分布 IID (均匀打乱)，特征分布 Non-IID (每个客户端应用不同变换)
    """
    os.makedirs(result_dir, exist_ok=True)
    
    client_data = {}
    client_class_distributions = {}
    
    # 1. 全局随机打乱数据 (IID 标签)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    # 2. 均匀切分索引
    client_indices_list = np.array_split(indices, num_clients)
    client_sizes = []
    
    # 用于日志显示的描述
    aug_descriptions = [
        "Original", "Brightness", "Contrast", "Rotation", 
        "Translate", "Shear", "Blur", "Sharpness", 
        "Posterize", "Invert"
    ]

    print(f"\n[Feature Shift - FashionMNIST] Starting data split...")

    for client_id in range(num_clients):
        # 获取切分后的索引
        indices = client_indices_list[client_id]
        client_sizes.append(len(indices))
        
        # 获取数据 (X shape: [N, 28, 28, 1])
        X_client = X[indices]
        Y_client = Y[indices]
        
        # 3. 获取特定的变换 (关键步骤)
        custom_transform = get_feature_shift_transform(client_id)
        
        # 创建数据集
        # 注意：这里使用的是你代码中定义的 FashionMNISTDataset
        dataset = FashionMNISTDataset(X_client, Y_client, transform=custom_transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_data[client_id] = train_loader
        
        # 记录分布
        counts_full = [0] * len(classes)
        unique, counts = np.unique(Y_client, return_counts=True)
        for u, c in zip(unique, counts):
            counts_full[int(u)] = int(c)
        client_class_distributions[client_id] = counts_full
        
        aug_name = aug_descriptions[client_id % len(aug_descriptions)]
        print(f"  Client {client_id} [{aug_name}]: Size {len(indices)}, Label Dist: {counts_full}")

    # 4. 保存日志
    with open(f"{result_dir}/client_distribution_feature_shift.txt", "w") as f:
        f.write(f"--- Feature Shift Non-IID Setup (Fashion-MNIST) ---\n")
        f.write(f"Global class distribution: {classes}\n\n")
        
        for client_id in range(num_clients):
            size = client_sizes[client_id]
            counts = client_class_distributions[client_id]
            aug_name = aug_descriptions[client_id % len(aug_descriptions)]
            f.write(f"Client {client_id} - {aug_name} - Size: {size}, Dist: {counts}\n")
    
    return client_data, client_class_distributions