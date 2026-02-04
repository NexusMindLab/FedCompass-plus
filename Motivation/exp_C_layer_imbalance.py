# exp_C_layer_imbalance.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import defaultdict
from data_utils import load_data, split_data_to_clients_IID, CIFAR10Dataset, get_transform
from modelx import HybridModelX

def setup_experiment_C():
    """设置实验C的参数和目录"""
    config = {
        "num_clients": 1,
        "batch_size": 32,
        "classes": [0, 1, 2, 3],
        "result_dir": "Result/exp_C",
        "alpha": 1.0,
        "input_channels": 3,
        "classical": "resnet18",
        "quantum": "qcnn",
        "num_epochs": 30, #30
        "learning_rate": 0.001,
        "moving_avg_window": 1,  # 滑动平均窗口
        "detailed_batches": 5,    # 记录详细梯度的batch数
    }
    os.makedirs(config["result_dir"], exist_ok=True)
    return config

class GradientStatistics:
    """梯度统计类 - 使用科学的方法论"""
    @staticmethod
    def compute_layer_statistics(grads):
        """计算单层的梯度统计"""
        if not grads:
            return None
        
        # 合并所有梯度张量
        all_grads = torch.cat([g.view(-1) for g in grads])
        
        # 基本统计
        stats_dict = {
            'norm': all_grads.norm().item(),              # L2范数
            'abs_mean': all_grads.abs().mean().item(),    # 绝对均值 - 衡量梯度幅度
            'mean': all_grads.mean().item(),              # 原始均值 - 衡量梯度方向偏差
            'std': all_grads.std().item(),                # 标准差
            'var': all_grads.var().item(),                # 方差
            'max': all_grads.max().item(),
            'min': all_grads.min().item(),
            'q25': torch.quantile(all_grads, 0.25).item(),
            'q75': torch.quantile(all_grads, 0.75).item(),
            'num_params': len(all_grads),                  # 参数数量
        }
        
        # 高级统计（如果样本足够）
        if len(all_grads) > 10:
            # 使用detach()将张量从计算图中分离，再转换为numpy
            grad_numpy = all_grads.detach().cpu().numpy()  # 修正：添加detach()
            stats_dict.update({
                'skewness': stats.skew(grad_numpy),        # 偏度
                'kurtosis': stats.kurtosis(grad_numpy),    # 峰度
            })
        
        return stats_dict
    
    @staticmethod
    def compute_relative_update(param, initial):
        """计算参数的相对更新幅度"""
        delta = param.data - initial
        # 相对变化：变化量 / (初始值 + epsilon)
        rel_change = delta / (initial.abs() + 1e-8)
        # 使用均方根变化率
        rms_change = torch.sqrt(torch.mean(rel_change ** 2)).item()
        return rms_change

class TrainingMonitor:
    """训练过程监控器（科学方法论版）"""
    def __init__(self, config):
        self.config = config
        
        # 梯度统计（按batch存储）
        self.classical_stats = []  # 每个元素是compute_layer_statistics返回的字典
        self.quantum_stats = []    # 每个元素是compute_layer_statistics返回的字典
        
        # 参数更新统计
        self.classical_updates = []  # 相对更新率
        self.quantum_updates = []    # 相对更新率
        
        # 训练过程统计
        self.losses = []
        self.accuracies = []
        self.learning_rates = []
        
        # 详细梯度分布（仅记录少量样本）
        self.detailed_gradients = {
            'classical': defaultdict(list),
            'quantum': defaultdict(list)
        }
        
        # 层间指标
        self.layer_metrics = []  # 存储每个batch的层间比较指标
        
    def record_batch_stats(self, classical_grads, quantum_grads, 
                          classical_params, quantum_params,
                          initial_params, loss, lr, batch_idx):
        """记录每个batch的统计信息"""
        
        # 1. 计算梯度统计
        c_stats = GradientStatistics.compute_layer_statistics(classical_grads)
        q_stats = GradientStatistics.compute_layer_statistics(quantum_grads)
        
        if c_stats:
            self.classical_stats.append(c_stats)
        if q_stats:
            self.quantum_stats.append(q_stats)
        
        # 2. 计算参数更新统计
        if classical_params and initial_params:
            c_updates = []
            for param in classical_params:
                if param.data_ptr() in initial_params:
                    update_rate = GradientStatistics.compute_relative_update(
                        param, initial_params[param.data_ptr()]
                    )
                    c_updates.append(update_rate)
            if c_updates:
                self.classical_updates.append(np.mean(c_updates))
        
        if quantum_params and initial_params:
            q_updates = []
            for param in quantum_params:
                if param.data_ptr() in initial_params:
                    update_rate = GradientStatistics.compute_relative_update(
                        param, initial_params[param.data_ptr()]
                    )
                    q_updates.append(update_rate)
            if q_updates:
                self.quantum_updates.append(np.mean(q_updates))
        
        # 3. 记录训练过程
        self.losses.append(loss)
        self.learning_rates.append(lr)
        
        # 4. 记录详细梯度（仅前几个batch）
        if batch_idx < self.config['detailed_batches']:
            if classical_grads:
                for g in classical_grads:
                    # 修正：使用detach()分离张量
                    self.detailed_gradients['classical'][batch_idx].extend(
                        g.detach().view(-1).cpu().numpy().tolist()
                    )
            if quantum_grads:
                for g in quantum_grads:
                    # 修正：使用detach()分离张量
                    self.detailed_gradients['quantum'][batch_idx].extend(
                        g.detach().view(-1).cpu().numpy().tolist()
                    )
        
        # 5. 计算层间指标
        if c_stats and q_stats:
            metrics = self._compute_layer_metrics(c_stats, q_stats)
            self.layer_metrics.append(metrics)
    
    def _compute_layer_metrics(self, c_stats, q_stats):
        """计算层间比较指标"""
        metrics = {}
        
        # 梯度尺度比率（归一化）
        metrics['grad_norm_ratio'] = c_stats['norm'] / (q_stats['norm'] + 1e-8)
        
        # 梯度幅度比率（使用绝对均值）
        metrics['abs_mean_ratio'] = c_stats['abs_mean'] / (q_stats['abs_mean'] + 1e-8)
        
        # 梯度方差比率（相对）
        metrics['var_ratio'] = q_stats['var'] / (c_stats['var'] + 1e-8)
        
        # 梯度分布范围比率
        c_range = c_stats['max'] - c_stats['min']
        q_range = q_stats['max'] - q_stats['min']
        metrics['range_ratio'] = q_range / (c_range + 1e-8)
        
        # 参数数量比率（用于归一化）
        metrics['param_ratio'] = q_stats['num_params'] / (c_stats['num_params'] + 1e-8)
        
        # 梯度漂移指标（均值的绝对值）
        metrics['classical_drift'] = abs(c_stats['mean'])
        metrics['quantum_drift'] = abs(q_stats['mean'])
        metrics['drift_ratio'] = metrics['quantum_drift'] / (metrics['classical_drift'] + 1e-8)
        
        return metrics
    
    def record_epoch_accuracy(self, accuracy):
        """记录每个epoch的准确率"""
        self.accuracies.append(accuracy)
    
    def get_summary_statistics(self):
        """获取训练过程摘要统计"""
        summary = {}
        
        # 梯度统计摘要
        if self.classical_stats and self.quantum_stats:
            # 计算平均统计
            def avg_stats(stat_list, key):
                values = [s[key] for s in stat_list if key in s]
                return np.mean(values) if values else 0.0
            
            summary.update({
                'c_norm_avg': avg_stats(self.classical_stats, 'norm'),
                'q_norm_avg': avg_stats(self.quantum_stats, 'norm'),
                'c_abs_mean_avg': avg_stats(self.classical_stats, 'abs_mean'),
                'q_abs_mean_avg': avg_stats(self.quantum_stats, 'abs_mean'),
                'c_var_avg': avg_stats(self.classical_stats, 'var'),
                'q_var_avg': avg_stats(self.quantum_stats, 'var'),
                'c_drift_avg': avg_stats(self.classical_stats, 'mean'),
                'q_drift_avg': avg_stats(self.quantum_stats, 'mean'),
            })
            
            # 计算层间比率
            summary['norm_ratio_avg'] = summary['c_norm_avg'] / (summary['q_norm_avg'] + 1e-8)
            summary['var_ratio_avg'] = summary['q_var_avg'] / (summary['c_var_avg'] + 1e-8)
            summary['drift_ratio_avg'] = abs(summary['q_drift_avg']) / (abs(summary['c_drift_avg']) + 1e-8)
        
        # 更新统计摘要
        if self.classical_updates:
            summary['c_update_avg'] = np.mean(self.classical_updates)
        if self.quantum_updates:
            summary['q_update_avg'] = np.mean(self.quantum_updates)
        
        # 训练统计摘要
        if self.losses:
            summary['final_loss'] = self.losses[-1]
            summary['min_loss'] = np.min(self.losses)
        if self.accuracies:
            summary['final_acc'] = self.accuracies[-1]
            summary['max_acc'] = np.max(self.accuracies)
        
        return summary

def extract_model_parameters(model):
    """提取模型的经典层和量子层参数"""
    classical_params = []
    quantum_params = []
    classical_grad_params = []
    quantum_grad_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'feature_extractor' in name:
                classical_params.append(param)
                if param.grad is not None:
                    classical_grad_params.append(param)
            elif 'q_params' in name:
                quantum_params.append(param)
                if param.grad is not None:
                    quantum_grad_params.append(param)
    
    return classical_params, quantum_params, classical_grad_params, quantum_grad_params

def compute_accuracy(model, data_loader, device):
    """计算模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def experiment_layer_imbalance():
    """主实验函数"""
    print("=" * 70)
    print("实验C（修正版）：科学的梯度分析方法论")
    print("=" * 70)
    
    # 设置实验
    config = setup_experiment_C()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data(classes=config["classes"])
    client_datasets = split_data_to_clients_IID(
        x_train, y_train, config["num_clients"], config["classes"], config["result_dir"]
    )
    
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 初始化模型
    model = HybridModelX(
        classical=config["classical"],
        quantum=config["quantum"],
        input_channels=config["input_channels"],
        device=device
    )
    
    train_loader = DataLoader(
        client_datasets[0],
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    criterion = nn.CrossEntropyLoss()
    monitor = TrainingMonitor(config)
    
    print("开始训练并监控层间动态...")
    print(f"{'Epoch':^6} | {'Loss':^8} | {'Acc':^6} | {'NormRatio':^10} | {'VarRatio':^10} | {'UpdateRatio':^10}")
    print("-" * 75)
    
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 提取参数
            classical_params, quantum_params, _, _ = extract_model_parameters(model)
            
            # 保存初始参数值
            initial_params = {}
            for param in classical_params + quantum_params:
                initial_params[param.data_ptr()] = param.data.clone()
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 提取梯度
            _, _, classical_grads, quantum_grads = extract_model_parameters(model)
            
            # 参数更新
            optimizer.step()
            
            # 记录统计信息
            current_lr = scheduler.get_last_lr()[0]
            monitor.record_batch_stats(
                classical_grads, quantum_grads,
                classical_params, quantum_params,
                initial_params,
                loss.item(), current_lr, batch_idx
            )
            
            epoch_loss += loss.item()
            batch_count += 1
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch准确率
        accuracy = compute_accuracy(model, test_loader, device)
        monitor.record_epoch_accuracy(accuracy)
        
        # 计算epoch的平均层间指标
        if monitor.layer_metrics:
            epoch_metrics = monitor.layer_metrics[-batch_count:]  # 当前epoch的指标
            avg_norm_ratio = np.mean([m['grad_norm_ratio'] for m in epoch_metrics])
            avg_var_ratio = np.mean([m['var_ratio'] for m in epoch_metrics])
            
            # 计算更新比率
            if len(monitor.classical_updates) >= batch_count and len(monitor.quantum_updates) >= batch_count:
                epoch_c_updates = monitor.classical_updates[-batch_count:]
                epoch_q_updates = monitor.quantum_updates[-batch_count:]
                avg_update_ratio = np.mean(epoch_q_updates) / (np.mean(epoch_c_updates) + 1e-8)
            else:
                avg_update_ratio = 0.0
        else:
            avg_norm_ratio = avg_var_ratio = avg_update_ratio = 0.0
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        
        print(f"{epoch+1:^6} | {avg_loss:^8.4f} | {accuracy:^6.4f} | "
              f"{avg_norm_ratio:^10.2f} | {avg_var_ratio:^10.2f} | {avg_update_ratio:^10.2f}")
    
    # 分析训练动态
    analyze_training_dynamics(config, monitor)
    # 新增：保存详细结果到文件
    save_detailed_results(config, monitor)

def analyze_training_dynamics(config, monitor):
    """分析训练动态"""
    print(f"\n{'='*70}")
    print("训练动态综合分析（科学方法论）")
    print(f"{'='*70}")
    
    # 获取摘要统计
    summary = monitor.get_summary_statistics()
    
    if summary:
        print("\n梯度统计摘要：")
        print(f"  经典层平均梯度范数: {summary.get('c_norm_avg', 0):.6e}")
        print(f"  量子层平均梯度范数: {summary.get('q_norm_avg', 0):.6e}")
        print(f"  梯度范数比率 (经典/量子): {summary.get('norm_ratio_avg', 0):.2f}x")
        
        print(f"\n  经典层平均梯度幅度: {summary.get('c_abs_mean_avg', 0):.6e}")
        print(f"  量子层平均梯度幅度: {summary.get('q_abs_mean_avg', 0):.6e}")
        print(f"  梯度幅度比率 (经典/量子): {summary.get('c_abs_mean_avg', 0)/(summary.get('q_abs_mean_avg', 0)+1e-8):.2f}x")
        
        print(f"\n  经典层平均梯度方差: {summary.get('c_var_avg', 0):.6e}")
        print(f"  量子层平均梯度方差: {summary.get('q_var_avg', 0):.6e}")
        print(f"  梯度方差比率 (量子/经典): {summary.get('var_ratio_avg', 0):.2f}x")
        
        print(f"\n  经典层梯度漂移: {summary.get('c_drift_avg', 0):.6e}")
        print(f"  量子层梯度漂移: {summary.get('q_drift_avg', 0):.6e}")
        print(f"  梯度漂移比率 (量子/经典): {summary.get('drift_ratio_avg', 0):.2f}x")
        
        print(f"\n参数更新统计：")
        if 'c_update_avg' in summary:
            print(f"  经典层平均相对更新率: {summary['c_update_avg']:.6f}")
        if 'q_update_avg' in summary:
            print(f"  量子层平均相对更新率: {summary['q_update_avg']:.6f}")
        
        print(f"\n训练结果：")
        print(f"  最终测试准确率: {summary.get('final_acc', 0):.4f}")
        print(f"  最高测试准确率: {summary.get('max_acc', 0):.4f}")
        print(f"  最终训练损失: {summary.get('final_loss', 0):.4f}")
        print(f"  最低训练损失: {summary.get('min_loss', 0):.4f}")
        
        # 诊断分析
        print(f"\n诊断分析：")
        norm_ratio = summary.get('norm_ratio_avg', 1.0)
        var_ratio = summary.get('var_ratio_avg', 1.0)
        drift_ratio = summary.get('drift_ratio_avg', 1.0)
        
        if norm_ratio > 10:
            print("  ⚠️  检测到显著的梯度尺度不平衡：经典层主导 (>10x)")
        elif norm_ratio > 5:
            print("  ⚠️  检测到中度梯度尺度不平衡：经典层优势 (>5x)")
        elif norm_ratio < 0.2:
            print("  ⚠️  检测到反向梯度尺度不平衡：量子层主导 (<0.2x)")
        else:
            print("  ✅ 梯度尺度相对平衡")
        
        if var_ratio > 5:
            print(f"  ⚠️  检测到量子层梯度高方差：{var_ratio:.1f}x经典层，训练稳定性受影响")
        elif var_ratio > 10:
            print(f"  🚨 严重的量子层梯度高方差：{var_ratio:.1f}x经典层，可能导致训练震荡")
        
        if drift_ratio > 5:
            print(f"  ⚠️  检测到量子层梯度漂移显著：{drift_ratio:.1f}x经典层")
    else:
        print("❌ 未检测到有效的梯度数据")
    
    # 生成可视化
    generate_training_plots(config, monitor)

def generate_training_plots(config, monitor):
    """生成训练过程可视化（修正版：自动适配数据长度）"""
    print("生成科学可视化图表...")
    
    # 创建综合可视化
    fig = plt.figure(figsize=(20, 16))
    
    # === 内部辅助函数：智能滑动平均 ===
    def smart_moving_average(data, target_window):
        if not data:
            return np.array([]), np.array([])
        
        data_arr = np.array(data)
        N = len(data_arr)
        
        # 如果数据太少，无法进行预设窗口的平滑，则使用较小的窗口或不平滑
        if N < target_window:
            effective_window = max(1, N // 5) # 动态调整窗口
        else:
            effective_window = target_window
            
        if effective_window <= 1:
            return np.arange(N), data_arr
            
        # 使用 mode='valid' 进行卷积
        kernel = np.ones(effective_window) / effective_window
        y_smooth = np.convolve(data_arr, kernel, mode='valid')
        
        # 强制 X 轴长度与 Y 轴对齐
        # valid 卷积从第 (window-1) 个点开始有值
        start_idx = effective_window - 1
        x_smooth = np.arange(start_idx, start_idx + len(y_smooth))
        
        return x_smooth, y_smooth

    # 1. 准备数据
    window = config['moving_avg_window']
    
    # 提取原始数据列表
    c_norms = [s['norm'] for s in monitor.classical_stats if 'norm' in s]
    q_norms = [s['norm'] for s in monitor.quantum_stats if 'norm' in s]
    c_vars = [s['var'] for s in monitor.classical_stats if 'var' in s]
    q_vars = [s['var'] for s in monitor.quantum_stats if 'var' in s]
    c_abs_means = [s['abs_mean'] for s in monitor.classical_stats if 'abs_mean' in s]
    q_abs_means = [s['abs_mean'] for s in monitor.quantum_stats if 'abs_mean' in s]

    # 子图1：梯度范数对比
    ax1 = plt.subplot(3, 3, 1)
    if c_norms and q_norms:
        cx, cy = smart_moving_average(c_norms, window)
        qx, qy = smart_moving_average(q_norms, window)
        
        ax1.plot(cx, cy, label='Classical', linewidth=2, alpha=0.8)
        ax1.plot(qx, qy, label='Quantum', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Gradient Norm (L2)')
        ax1.set_title(f'Gradient Norms')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

    # 子图2：梯度幅度对比
    ax2 = plt.subplot(3, 3, 2)
    if c_abs_means and q_abs_means:
        cx, cy = smart_moving_average(c_abs_means, window)
        qx, qy = smart_moving_average(q_abs_means, window)
        
        ax2.plot(cx, cy, label='Classical', linewidth=2, alpha=0.8)
        ax2.plot(qx, qy, label='Quantum', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Absolute Gradient Mean')
        ax2.set_title('Gradient Magnitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    # 子图3：梯度方差对比
    ax3 = plt.subplot(3, 3, 3)
    if c_vars and q_vars:
        cx, cy = smart_moving_average(c_vars, window)
        qx, qy = smart_moving_average(q_vars, window)
        
        ax3.plot(cx, cy, label='Classical', linewidth=2, alpha=0.8)
        ax3.plot(qx, qy, label='Quantum', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Gradient Variance')
        ax3.set_title('Gradient Variance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

    # 子图4：层间比率变化
    ax4 = plt.subplot(3, 3, 4)
    if monitor.layer_metrics:
        norm_ratios = [m['grad_norm_ratio'] for m in monitor.layer_metrics]
        var_ratios = [m['var_ratio'] for m in monitor.layer_metrics]
        
        nx, ny = smart_moving_average(norm_ratios, window)
        vx, vy = smart_moving_average(var_ratios, window)
        
        ax4.plot(nx, ny, label='Norm Ratio (C/Q)', linewidth=2, color='blue')
        ax4.plot(vx, vy, label='Var Ratio (Q/C)', linewidth=2, color='red')
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Ratio')
        ax4.set_title('Layer-wise Gradient Ratios')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

    # 子图5：参数更新率对比
    ax5 = plt.subplot(3, 3, 5)
    if monitor.classical_updates and monitor.quantum_updates:
        cx, cy = smart_moving_average(monitor.classical_updates, window)
        qx, qy = smart_moving_average(monitor.quantum_updates, window)
        
        ax5.plot(cx, cy, label='Classical', linewidth=2)
        ax5.plot(qx, qy, label='Quantum', linewidth=2)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Relative Update Rate')
        ax5.set_title('Parameter Update Rates')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')

    # 子图6：准确率和损失曲线
    ax6 = plt.subplot(3, 3, 6)
    if monitor.accuracies:
        epochs = range(1, len(monitor.accuracies) + 1)
        ax6.plot(epochs, monitor.accuracies, marker='o', color='green', linewidth=2, label='Accuracy')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Accuracy', color='green')
        ax6.set_title('Test Accuracy')
        ax6.grid(True, alpha=0.3)
        
        ax6_loss = ax6.twinx()
        # 确保损失长度匹配
        if len(monitor.losses) > 1:
            loss_epochs = np.linspace(1, len(monitor.accuracies), len(monitor.losses))
            ax6_loss.plot(loss_epochs, monitor.losses, color='red', alpha=0.7, linewidth=1, label='Loss')
        ax6_loss.set_ylabel('Loss', color='red')

    # 子图7：梯度分布直方图（第一个batch）
    ax7 = plt.subplot(3, 3, 7)
    if monitor.detailed_gradients['classical'] and monitor.detailed_gradients['quantum']:
        batch_idx = 0
        if batch_idx in monitor.detailed_gradients['classical']:
            c_grads = monitor.detailed_gradients['classical'][batch_idx]
            q_grads = monitor.detailed_gradients['quantum'][batch_idx]
            
            all_grads = c_grads + q_grads
            if all_grads: # 确保非空
                # 过滤掉极端异常值以获得更好的直方图显示
                lower, upper = np.percentile(all_grads, 1), np.percentile(all_grads, 99)
                bins = np.linspace(lower, upper, 50)
                
                ax7.hist(c_grads, bins=bins, alpha=0.6, label='Classical', density=True, 
                        color='blue', edgecolor='black', linewidth=0.5)
                ax7.hist(q_grads, bins=bins, alpha=0.6, label='Quantum', density=True,
                        color='orange', edgecolor='black', linewidth=0.5)
                ax7.set_xlabel('Gradient Value')
                ax7.set_ylabel('Density')
                ax7.set_title(f'Gradient Distribution (Batch {batch_idx})')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
                ax7.set_yscale('log')

    # 子图8：梯度分布箱线图
    ax8 = plt.subplot(3, 3, 8)
    if monitor.detailed_gradients['classical'] and monitor.detailed_gradients['quantum']:
        c_all_grads = []
        q_all_grads = []
        for batch_idx in range(min(5, len(monitor.detailed_gradients['classical']))):
            if batch_idx in monitor.detailed_gradients['classical']:
                c_all_grads.extend(monitor.detailed_gradients['classical'][batch_idx])
            if batch_idx in monitor.detailed_gradients['quantum']:
                q_all_grads.extend(monitor.detailed_gradients['quantum'][batch_idx])
        
        if c_all_grads and q_all_grads:
            # 随机采样以避免绘图过慢
            sample_size = 2000
            c_sample = np.random.choice(c_all_grads, min(len(c_all_grads), sample_size), replace=False)
            q_sample = np.random.choice(q_all_grads, min(len(q_all_grads), sample_size), replace=False)
            
            box_data = [c_sample, q_sample]
            box = ax8.boxplot(box_data, labels=['Classical', 'Quantum'], patch_artist=True)
            
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            ax8.set_ylabel('Gradient Value')
            ax8.set_title('Gradient Distribution Boxplot')
            ax8.grid(True, alpha=0.3)

    # 子图9：梯度统计散点图（均值vs方差）
    ax9 = plt.subplot(3, 3, 9)
    if monitor.classical_stats and monitor.quantum_stats:
        sample_step = max(1, len(monitor.classical_stats) // 200) # 限制点数
        c_means = [s['mean'] for s in monitor.classical_stats[::sample_step]]
        c_vars = [s['var'] for s in monitor.classical_stats[::sample_step]]
        q_means = [s['mean'] for s in monitor.quantum_stats[::sample_step]]
        q_vars = [s['var'] for s in monitor.quantum_stats[::sample_step]]
        
        ax9.scatter(c_means, c_vars, alpha=0.5, label='Classical', s=20, color='blue')
        ax9.scatter(q_means, q_vars, alpha=0.5, label='Quantum', s=20, color='orange')
        ax9.set_xlabel('Gradient Mean')
        ax9.set_ylabel('Gradient Variance')
        ax9.set_title('Mean-Variance Phase Plot')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_xscale('symlog', linthresh=1e-8)
        ax9.set_yscale('log')

    plt.tight_layout()
    try:
        plt.savefig(f"{config['result_dir']}/scientific_analysis.png", dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {config['result_dir']}/scientific_analysis.png")
    except Exception as e:
        print(f"保存图表失败: {str(e)}")
    finally:
        plt.close()

def save_detailed_results(config, monitor):
    """保存详细实验结果"""
    results_file = f"{config['result_dir']}/detailed_results.txt"
    
    with open(results_file, "w") as f:
        f.write("实验C（修正版）：经典层与量子层训练动态分析\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("配置参数:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n训练摘要统计:\n")
        summary = monitor.get_summary_statistics()
        for key, value in summary.items():
            f.write(f"  {key}: {value:.6e}\n")
        
        f.write("\n层间失衡诊断:\n")
        norm_ratio = summary.get('norm_ratio_avg', 1.0)
        var_ratio = summary.get('var_ratio_avg', 1.0)
        
        if norm_ratio > 10:
            f.write("  严重梯度尺度失衡：经典层主导 (>10x)\n")
        elif norm_ratio > 5:
            f.write("  中度梯度尺度失衡：经典层优势 (>5x)\n")
        elif norm_ratio < 0.2:
            f.write("  反向梯度尺度失衡：量子层主导 (<0.2x)\n")
        else:
            f.write("  梯度尺度相对平衡\n")
        
        if var_ratio > 10:
            f.write(f"  严重量子层梯度高方差：{var_ratio:.1f}x经典层\n")
        elif var_ratio > 5:
            f.write(f"  量子层梯度高方差：{var_ratio:.1f}x经典层\n")
        else:
            f.write("  梯度方差相对平衡\n")
        
        if len(monitor.accuracies) > 0:
            f.write(f"\n最终测试准确率: {monitor.accuracies[-1]:.4f}\n")
            f.write(f"最高测试准确率: {np.max(monitor.accuracies):.4f}\n")
        
        if len(monitor.losses) > 0:
            f.write(f"最终训练损失: {monitor.losses[-1]:.4f}\n")
            f.write(f"最低训练损失: {np.min(monitor.losses):.4f}\n")
    
    print(f"详细结果已保存至: {results_file}")

if __name__ == "__main__":
    experiment_layer_imbalance()