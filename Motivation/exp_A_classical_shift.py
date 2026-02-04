# exp_A_classical_shift.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from data_utils_A import load_data, split_data_to_clients, CIFAR10Dataset, get_transform
from modelx import HybridModelX

def setup_experiment_A():
    """设置实验A的参数和目录"""
    config = {
        "num_clients": 10,
        "batch_size": 32,
        "classes": [0, 1, 2, 3],
        "result_dir": "Result/exp_A",
        "alpha_values": [1.0, 0.3],  # IID和高度非IID
        'sample_per_client': 200,  # 每个客户端用于统计特征的样本数（默认200）
        'sample_batch_size': 128,  # 每个采样batch的大小
        'hist_bins': 50,  # 直方图bins数，用于JS距离计算
        "input_channels": 3,
        "classical": "resnet18",
        "quantum": "qcnn",
        "num_rounds": 5,
        "local_epochs": 5
    }
    
    # 创建结果目录
    for alpha in config["alpha_values"]:
        os.makedirs(f"{config['result_dir']}/alpha_{alpha}", exist_ok=True)
    
    return config

def train_fedavg_simulation(config, alpha, device):
    """在指定alpha下进行简化的FedAvg训练模拟"""
    print(f"正在为α={alpha}进行FedAvg训练模拟...")
    
    # 加载数据
    train_dataset, test_dataset = load_data(alpha=alpha, classes=config["classes"], transform=get_transform())
    client_datasets = split_data_to_clients(train_dataset, num_clients=config["num_clients"], result_dir=f"{config['result_dir']}/alpha_{alpha}", iid=(alpha == 1.0))
    
    # 初始化模型（全局 + 客户端本地副本）
    global_model = HybridModelX(input_channels=config["input_channels"], classical=config["classical"], quantum=config["quantum"])
    client_models = []
    for i in range(config["num_clients"]):
        m = HybridModelX(input_channels=config["input_channels"], classical=config["classical"], quantum=config["quantum"])
        m.load_state_dict(global_model.state_dict())
        client_models.append(m)
    
    # 进行若干轮简单联邦训练（每轮所有客户端都参与，使用本地 SGD）
    for rnd in range(config["num_rounds"]):
        print(f"[Round {rnd+1}/{config['num_rounds']}]")
        for client_id, client_model in enumerate(client_models):
            client_model.train()
            # 创建 DataLoader
            train_loader = DataLoader(client_datasets[client_id], batch_size=config["batch_size"], shuffle=True)
            optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # 本地训练多个epoch
            for epoch in range(config["local_epochs"]):
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = client_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
        # 聚合（简单平均）
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            # 如果是量子层参数（假设以 "quantum" 在 key 中为标识），使用简单平均
            if "quantum" in key:
                quantum_params = [client_models[i].state_dict()[key].float() for i in range(len(client_models))]
                global_dict[key] = torch.mean(torch.stack(quantum_params), dim=0)
            else:
                # 对经典参数使用加权平均，确保使用浮点数
                client_sizes = [len(client_datasets[i]) for i in range(config["num_clients"])]
                total_size = sum(client_sizes)
                weighted_sum = torch.zeros_like(global_dict[key].float())
                
                for i, client_model in enumerate(client_models):
                    weight = client_sizes[i] / total_size
                    # 确保参数转换为浮点数
                    param_float = client_model.state_dict()[key].float()
                    weighted_sum += param_float * weight
                
                # 保持原始数据类型
                global_dict[key] = weighted_sum.to(global_dict[key].dtype)
        
        global_model.load_state_dict(global_dict)
        # 把聚合后的全局模型下发到每个客户端
        for m in client_models:
            m.load_state_dict(global_model.state_dict())
    
    # 在训练完成后，返回全局模型和每个客户端模型、数据集、测试集
    return global_model, client_models, client_datasets, test_dataset.images, test_dataset.labels

def analyze_feature_distribution(config, alpha, global_model, client_models, client_datasets, device):
    """分析特征分布"""
    print(f"分析α={alpha}的特征分布...")
    
    # 收集所有客户端的特征向量
    all_features = []
    all_client_labels = []
    all_data_labels = []
    
    with torch.no_grad():
        for client_id, client_model in enumerate(client_models):
            client_model.eval()
            
            # 从该客户端数据集中采样（支持采集多batch直到达到 sample_per_client）
            sample_batch = min(config.get('sample_batch_size', 128), len(client_datasets[client_id]))
            sample_loader = DataLoader(
                client_datasets[client_id],
                batch_size=sample_batch,
                shuffle=True
            )

            collected = 0
            for images, labels in sample_loader:
                images = images.to(device)
                # 提取经典特征（在量子层之前）
                features = client_model.feature_extractor(images)
                feat_np = features.cpu().numpy()
                need = config.get('sample_per_client', 200) - collected
                if need <= 0:
                    break
                if feat_np.shape[0] > need:
                    feat_np = feat_np[:need]
                    labels_np = labels.numpy()[:need]
                else:
                    labels_np = labels.numpy()
                all_features.append(feat_np)
                all_client_labels.extend([client_id] * feat_np.shape[0])
                all_data_labels.extend(labels_np.tolist())
                collected += feat_np.shape[0]
                if collected >= config.get('sample_per_client', 200):
                    break

    # 合并所有特征
    all_features = np.vstack(all_features)
    all_client_labels = np.array(all_client_labels)
    all_data_labels = np.array(all_data_labels)
    
    print(f"特征矩阵形状: {all_features.shape}")
    print(f"客户端标签分布: {np.unique(all_client_labels, return_counts=True)}")
    print(f"数据类别分布: {np.unique(all_data_labels, return_counts=True)}")
    
    # 计算客户端间特征分布距离
    feature_distances = calculate_feature_distribution_distance(all_features, all_client_labels, config)
    
    # 计算每个客户端特征中心与总体中心
    client_centers = []
    for cid in np.unique(all_client_labels):
        mask = all_client_labels == cid
        center = all_features[mask].mean(axis=0)
        client_centers.append(center)
    client_centers = np.vstack(client_centers)
    
    # 特征维度方差
    feature_variance = np.var(all_features, axis=0)
    
    # 可视化（t-SNE / PCA / 客户端中心 / 距离矩阵 / 方差直方图）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(all_features)
    
    # 使用t-SNE降维，perplexity 更稳健的选择
    n_samples = len(all_features)
    perp = min(30, max(2, n_samples // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    features_2d = tsne.fit_transform(all_features)
    
    # 1. t-SNE按客户端着色
    scatter1 = axes[0, 0].scatter(features_2d[:, 0], features_2d[:, 1], c=all_client_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_title(f't-SNE by Client (α={alpha})')
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Client ID')
    
    # 2. t-SNE按数据类别着色
    scatter2 = axes[0, 1].scatter(features_2d[:, 0], features_2d[:, 1], c=all_data_labels, cmap='viridis', alpha=0.7)
    axes[0, 1].set_title(f't-SNE by Class (α={alpha})')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Class ID')
    
    # 3. PCA按客户端着色
    scatter3 = axes[0, 2].scatter(features_pca[:, 0], features_pca[:, 1], c=all_client_labels, cmap='tab10', alpha=0.7)
    axes[0, 2].set_title(f'PCA by Client (α={alpha})')
    axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)')
    axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)')
    plt.colorbar(scatter3, ax=axes[0, 2], label='Client ID')
    
    # 4. 客户端特征中心
    for i, c in enumerate(client_centers):
        axes[1, 0].scatter(c[0], c[1], s=200, label=f'C{i}')
        axes[1, 0].text(c[0], c[1], f'C{i}', fontsize=9)
    axes[1, 0].set_title(f'Client Feature Centers (α={alpha})')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    axes[1, 0].legend(loc='best', fontsize=8)
    
    # 5. 客户端间距离矩阵可视化（用 feature_distances 填充一个对称矩阵）
    nclients = len(np.unique(all_client_labels))
    dist_mat = np.zeros((nclients, nclients))
    for key, val in feature_distances.items():
        # key 格式: client_i_vs_j
        parts = key.split('_')
        i = int(parts[1])
        j = int(parts[3])
        dist_mat[i, j] = val
        dist_mat[j, i] = val
    im = axes[1, 1].imshow(dist_mat, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title(f'Inter-Client Distance Matrix (α={alpha})')
    axes[1, 1].set_xlabel('Client ID')
    axes[1, 1].set_ylabel('Client ID')
    plt.colorbar(im, ax=axes[1, 1], label='Distance')
    
    # 6. 特征方差分布直方图
    axes[1, 2].hist(feature_variance, bins=30, alpha=0.6, edgecolor='k')
    axes[1, 2].axvline(feature_variance.mean(), color='r', linestyle='--', label=f"Mean: {feature_variance.mean():.4f}")
    axes[1, 2].set_title(f'Feature Variance Distribution (α={alpha})')
    axes[1, 2].set_xlabel('Variance')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{config['result_dir']}/alpha_{alpha}/feature_analysis.pdf", 
                bbox_inches='tight')
    plt.close()
    
    # 计算量化指标
    center_variance = np.var(client_centers, axis=0).sum() if len(client_centers) > 0 else 0
    avg_feature_variance = feature_variance.mean()
    
    return center_variance, avg_feature_variance, feature_distances

def calculate_feature_distribution_distance(features, client_labels, config=None):
    """计算客户端间特征分布的统计距离（基于一维直方图的 JS 散度）。
    features: numpy array shape (N, D) 或 list of arrays; client_labels: array-like length N 或 list aligned with features
    config: 用于覆盖 hist_bins 等参数的字典
    """
    from scipy.spatial.distance import jensenshannon
    eps = 1e-12
    bins = 50 if config is None else config.get('hist_bins', 50)

    # 如果 inputs 是 list of arrays，将其合并为统一的 numpy array
    if isinstance(features, list):
        # 每个元素可能是 shape (n_i, d)，合并后变为 (sum n_i, d)
        features_arr = np.vstack(features)
    else:
        features_arr = np.asarray(features)

    client_labels = np.asarray(client_labels)
    client_ids = np.unique(client_labels)
    feature_distances = {}

    # 按 client 合并特征
    client_feature_map = {}
    for cid in client_ids:
        mask = client_labels == cid
        client_feature_map[int(cid)] = features_arr[mask]

    for i in range(len(client_ids)):
        for j in range(i+1, len(client_ids)):
            ci = int(client_ids[i]); cj = int(client_ids[j])
            feats_i = client_feature_map[ci].flatten()
            feats_j = client_feature_map[cj].flatten()
            if feats_i.size == 0 or feats_j.size == 0:
                dist = 0.0
            else:
                combined_min = min(feats_i.min(), feats_j.min())
                combined_max = max(feats_i.max(), feats_j.max())
                if combined_min == combined_max:
                    dist = 0.0
                else:
                    hist_i, _ = np.histogram(feats_i, bins=bins, range=(combined_min, combined_max), density=True)
                    hist_j, _ = np.histogram(feats_j, bins=bins, range=(combined_min, combined_max), density=True)
                    hist_i = hist_i + eps
                    hist_j = hist_j + eps
                    hist_i = hist_i / np.sum(hist_i)
                    hist_j = hist_j / np.sum(hist_j)
                    dist = float(jensenshannon(hist_i, hist_j))
            feature_distances[f'client_{ci}_vs_{cj}'] = dist

    return feature_distances

def evaluate_model_performance(config, global_model, client_models, x_test, y_test, device):
    """评估全局模型和客户端模型在测试集上的性能"""
    # 创建测试集DataLoader
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 评估全局模型
    global_model.eval()
    correct_global, total_global = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_global += labels.size(0)
            correct_global += (predicted == labels).sum().item()
    
    global_accuracy = 100 * correct_global / total_global
    
    # 评估各客户端模型
    client_accuracies = []
    for client_id, client_model in enumerate(client_models):
        client_model.eval()
        correct_client, total_client = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = client_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_client += labels.size(0)
                correct_client += (predicted == labels).sum().item()
        
        client_accuracy = 100 * correct_client / total_client
        client_accuracies.append(client_accuracy)
    
    return global_accuracy, client_accuracies

def analyze_correlation(feature_metrics, performance_metrics, config):
    """分析特征分布指标与模型性能指标的相关性"""
    center_variances = [result['center_variance'] for result in feature_metrics]
    performance_drops = [result['performance_drop'] for result in performance_metrics]
    
    # 计算相关系数
    corr = np.corrcoef(center_variances, performance_drops)[0, 1]
    print(f"特征分布方差与性能下降的皮尔逊相关系数: {corr:.4f}")
    return corr

def experiment_classical_shift():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = setup_experiment_A()
    
    results = []
    for alpha in config["alpha_values"]:
        print(f"\n{'='*50}")
        print(f"处理 α = {alpha}")
        print(f"{'='*50}")
        
        # 训练模型
        global_model, client_models, client_datasets, x_test, y_test = train_fedavg_simulation(config, alpha, device)
        
        # 分析特征分布 - 修改这里：只接收3个返回值
        center_variance, avg_feature_variance, feature_distances = analyze_feature_distribution(
            config, alpha, global_model, client_models, client_datasets, device
        )
        
        # 评估性能
        global_acc, client_accs = evaluate_model_performance(config, global_model, client_models, x_test, y_test, device)
        
        res = {
            "alpha": alpha,
            "center_variance": center_variance,
            "avg_feature_variance": avg_feature_variance,
            "feature_distances": feature_distances,
            "global_accuracy": global_acc,
            "client_accuracies": client_accs
        }
        results.append(res)
    
    # 打印和保存结果
    iid_result = next(r for r in results if r["alpha"] == 1.0)
    non_iid_result = next(r for r in results if r["alpha"] == 0.3)
    
    center_variance_ratio = non_iid_result["center_variance"] / max(1e-12, iid_result["center_variance"])
    performance_drop = iid_result["global_accuracy"] - non_iid_result["global_accuracy"]
    
    with open(f"{config['result_dir']}/report.txt", "w", encoding="utf-8") as f:
        f.write("实验A: 客户端间特征分布差异分析\n")
        f.write("="*50 + "\n\n")
        f.write("配置参数:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n结果对比:\n")
        f.write(f"客户端特征中心方差:\n  IID (α=1.0): {iid_result['center_variance']:.6f}\n  非IID (α=0.3): {non_iid_result['center_variance']:.6f}\n  方差倍数: {center_variance_ratio:.2f}x\n\n")
        f.write(f"特征空间离散度:\n  IID (α=1.0): {iid_result['avg_feature_variance']:.6f}\n  非IID (α=0.3): {non_iid_result['avg_feature_variance']:.6f}\n  离散度倍数: {non_iid_result['avg_feature_variance']/max(1e-12,iid_result['avg_feature_variance']):.2f}x\n\n")
        f.write(f"模型性能:\n  IID (α=1.0): {iid_result['global_accuracy']:.2f}%\n  非IID (α=0.3): {non_iid_result['global_accuracy']:.2f}%\n  性能下降: {performance_drop:.2f}%\n\n")
        f.write("客户端间特征分布距离（示例，均为 JS 散度）:\n")
        f.write("  IID (α=1.0):\n")
        for pair, distance in iid_result['feature_distances'].items():
            f.write(f"    {pair}: {distance:.4f}\n")
        f.write(f"  非IID (α=0.3):\n")
        for pair, distance in non_iid_result['feature_distances'].items():
            f.write(f"    {pair}: {distance:.4f}\n")
        
        if center_variance_ratio > 1.2 and performance_drop > 2.0:
            f.write("\n结论: 非IID数据导致客户端特征提取层出现显著偏移，并引起模型性能下降\n")
        else:
            f.write("\n结论: 数据分布对特征提取层影响较小\n")
    
    print(f"\n实验完成! 结果保存在: {config['result_dir']}")

if __name__ == "__main__":
    experiment_classical_shift()
