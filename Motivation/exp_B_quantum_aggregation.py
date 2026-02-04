# exp_B_quantum_aggregation.py
import os
import random
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_utils import load_data, split_data_to_clients, CIFAR10Dataset, get_transform
from modely import HybridModelX
from FL import HybridClient, FedAvgHybrid, server_evaluate
import matplotlib.pyplot as plt
from scipy import stats

# 配置参数 - 挑战B实验专用配置
FL_CONFIG = {
    "server_round": 5,  # 增加轮数以更好观察收敛稳定性
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "result_dir": "Result/exp_B", 
    "alpha": 0.7,              # 使用轻度非IID以专注于量子层问题
    "classical": "resnet18",
    "quantum": "qcnn",
    "input_channels": 3,
    "quantum_aggregation": "arithmetic"  # 实验变量：arithmetic 或 circular
}

def enhanced_circular_mean_aggregation(parameters_list, weights, param_names):
    """增强的圆周平均聚合函数 - 基于参数名称精确识别量子参数"""
    # 调试信息：添加在这里
    print(f"[DEBUG] Total parameters: {len(param_names)}")
    print(f"[DEBUG] Quantum parameters found: {sum(1 for name in param_names if name == 'q_params')}")
    aggregated_params = []
    
    for param_idx, param_name in enumerate(param_names):
        # 精确识别量子参数 - 只识别 'q_params'
        is_quantum_param = (param_name == 'q_params')
        
        if is_quantum_param:
            print(f"[DEBUG] Applying circular mean to quantum parameter: {param_name}")
            # 对量子参数使用圆周平均
            angles_list = []
            weight_list = []
            
            for client_idx, params in enumerate(parameters_list):
                client_angles = params[param_idx].flatten()
                client_weight = weights[client_idx] / len(client_angles)
                
                angles_list.extend(client_angles)
                weight_list.extend([client_weight] * len(client_angles))
            
            # 计算加权圆周平均
            sin_sum = sum(w * np.sin(angle) for angle, w in zip(angles_list, weight_list))
            cos_sum = sum(w * np.cos(angle) for angle, w in zip(angles_list, weight_list))
            
            aggregated_angle = np.arctan2(sin_sum, cos_sum)
            # 确保角度在 [0, 2π) 范围内
            aggregated_angle = aggregated_angle % (2 * np.pi)
            
            aggregated_param = np.full_like(parameters_list[0][param_idx], aggregated_angle)
            
        else:
            # 对经典参数使用算术平均
            aggregated_param = np.zeros_like(parameters_list[0][param_idx], dtype=np.float64)
            total_weight = 0.0
            
            for client_idx, params in enumerate(parameters_list):
                weight = weights[client_idx]
                aggregated_param += weight * params[param_idx]
                total_weight += weight
            
            if total_weight > 0:
                aggregated_param /= total_weight
                
        aggregated_params.append(aggregated_param)
    
    return aggregated_params

def demonstrate_period_mismatch():
    """展示周期性参数不匹配的具体例子 - 修复版本"""
    # 模拟两个客户端的量子参数
    client1_angles = np.array([0.1, 0.2])  # 接近0的角度
    client2_angles = np.array([0.1 + 2*np.pi, 0.2 + 2*np.pi])  # 等效角度，但数值差异大
    
    # 算术平均
    arithmetic_mean = (client1_angles + client2_angles) / 2  # 得到[π+0.1, π+0.2]
    
    # 圆周平均 - 修正实现
    sin_mean = (np.sin(client1_angles) + np.sin(client2_angles)) / 2
    cos_mean = (np.cos(client1_angles) + np.cos(client2_angles)) / 2
    circular_mean = np.arctan2(sin_mean, cos_mean)
    # 确保角度在 [0, 2π) 范围内
    circular_mean = circular_mean % (2 * np.pi)
    
    print("周期性冲突演示:")
    print(f"客户端1角度: {client1_angles}")
    print(f"客户端2角度: {client2_angles}") 
    print(f"算术平均结果: {arithmetic_mean} ← 错误!")
    print(f"圆周平均结果: {circular_mean} ← 正确!")
    
    # 可视化展示
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    angles = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(angles), np.sin(angles), 'k-', alpha=0.3)
    plt.scatter(np.cos(client1_angles), np.sin(client1_angles), c='red', s=100, label='Client 1')
    plt.scatter(np.cos(client2_angles), np.sin(client2_angles), c='blue', s=100, label='Client 2')
    plt.scatter(np.cos(arithmetic_mean), np.sin(arithmetic_mean), c='green', s=150, marker='*', label='Arithmetic Mean')
    plt.axis('equal')
    plt.title('Arithmetic Mean - Period Mismatch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cos(angles), np.sin(angles), 'k-', alpha=0.3)
    plt.scatter(np.cos(client1_angles), np.sin(client1_angles), c='red', s=100, label='Client 1')
    plt.scatter(np.cos(client2_angles), np.sin(client2_angles), c='blue', s=100, label='Client 2')
    plt.scatter(np.cos(circular_mean), np.sin(circular_mean), c='purple', s=150, marker='*', label='Circular Mean')
    plt.axis('equal')
    plt.title('Circular Mean - Correct Aggregation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{FL_CONFIG['result_dir']}/period_mismatch_demonstration.pdf", bbox_inches='tight')
    plt.close()
    
    return arithmetic_mean, circular_mean

def analyze_quantum_periodicity(quantum_params_history, method_name):
    """分析量子参数的周期性行为 - 修复版本"""
    if not quantum_params_history:
        return [], []
    
    print(f"[DEBUG] Analyzing quantum periodicity for {method_name}, rounds: {len(quantum_params_history)}")

    periodicity_conflicts = []
    round_variance = []
    
    for round_idx, round_params in enumerate(quantum_params_history):
        # 只分析真正的量子参数 'q_params'
        client_means = []
        for client_params in round_params:
            if client_params:  # 确保参数列表不为空
                # 只取第一个参数（q_params）
                param = client_params[0] if len(client_params) > 0 else None
                if param is not None and isinstance(param, np.ndarray) and param.size > 0:
                    client_means.append(np.mean(np.abs(param)))
        
        # 计算方差
        if client_means:
            round_var = np.var(client_means)
        else:
            round_var = 0.0
            
        round_variance.append(round_var)
        
        # 分析周期性冲突 - 只分析第一个量子参数
        if len(round_params) >= 2:
            client_angles = []
            for client_params in round_params:
                if client_params and len(client_params) > 0:
                    first_param = client_params[0]  # q_params
                    if isinstance(first_param, np.ndarray) and first_param.size > 0:
                        flattened = first_param.flatten()
                        client_angles.append(flattened)
            
            # 分析前几个参数维度
            if len(client_angles) >= 2:
                min_dims = min(len(angles) for angles in client_angles)
                
                for param_dim in range(min(min_dims, 10)):  # 分析前10个参数
                    angles_at_dim = [angles[param_dim] % (2 * np.pi) for angles in client_angles]
                    
                    # 检测潜在的周期性冲突
                    for i in range(len(angles_at_dim)):
                        for j in range(i+1, len(angles_at_dim)):
                            angle_i = angles_at_dim[i]
                            angle_j = angles_at_dim[j]
                            
                            circular_dist = min(
                                abs(angle_i - angle_j),
                                2 * np.pi - abs(angle_i - angle_j)
                            )
                            linear_dist = abs(angle_i - angle_j)
                            
                            # 如果圆周距离小但线性距离大，说明存在周期性冲突
                            if circular_dist < 0.5 and linear_dist > np.pi:  # 调整阈值
                                periodicity_conflicts.append({
                                    'round': round_idx,
                                    'method': method_name,
                                    'param_dim': param_dim,
                                    'client_pair': (i, j),
                                    'angles': (angle_i, angle_j),
                                    'circular_dist': circular_dist,
                                    'linear_dist': linear_dist,
                                    'conflict_severity': linear_dist - circular_dist
                                })
    
    # 保存周期性分析结果
    if periodicity_conflicts:
        result_dir = FL_CONFIG["result_dir"]
        os.makedirs(result_dir, exist_ok=True)
        with open(f"{result_dir}/periodicity_conflicts_{method_name}.txt", "w") as f:
            f.write(f"Periodicity Conflict Analysis - {method_name}\n")
            f.write("=" * 60 + "\n")
            for conflict in periodicity_conflicts:
                # 修复这里：分别格式化元组中的每个角度值
                angle1, angle2 = conflict['angles']
                f.write(f"Round {conflict['round']}, Dim {conflict['param_dim']}: "
                       f"Clients {conflict['client_pair']}, Angles ({angle1:.3f}, {angle2:.3f}), "
                       f"Conflict Severity: {conflict['conflict_severity']:.3f}\n")
    
    return periodicity_conflicts, round_variance

def calculate_training_stability(accuracy_history, loss_history, quantum_variance_history):
    """计算训练稳定性指标"""
    if len(accuracy_history) < 2:
        return {'stability_score': 0, 'accuracy_std': 0, 'loss_std': 0, 'loss_oscillations': 0}
    
    accuracy_std = np.std(accuracy_history)
    loss_std = np.std(loss_history)
    
    # 计算性能震荡次数（损失连续增加）
    loss_oscillations = 0
    for i in range(1, len(loss_history)):
        if loss_history[i] > loss_history[i-1]:
            loss_oscillations += 1
    
    # 计算量子参数稳定性
    quantum_stability = 1.0 / (np.mean(quantum_variance_history) + 1e-8) if quantum_variance_history else 1.0
    
    # 综合稳定性评分
    stability_score = 1.0 / (accuracy_std + loss_std + 0.1 * loss_oscillations + 1e-8)
    stability_score *= quantum_stability  # 结合量子参数稳定性
    
    return {
        'stability_score': stability_score,
        'accuracy_std': accuracy_std,
        'loss_std': loss_std, 
        'loss_oscillations': loss_oscillations,
        'quantum_stability': quantum_stability
    }

class ChallengeBStrategy(fl.server.strategy.FedAvg):
    """挑战B实验专用策略 - 修复版本"""
    
    def __init__(self, initial_parameters, evaluate_fn, param_names, aggregation_method="arithmetic", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.param_names = param_names
        self.aggregation_method = aggregation_method
        self.accuracy_history = []
        self.loss_history = []
        self.quantum_param_variance = []
        self.quantum_params_history = []
        
    def aggregate_fit(self, server_round, results, failures):
        """重写聚合方法以支持不同量子参数聚合策略 - 修复版本"""
        if not results:
            return None, {}
        
        # 计算权重（基于样本数量）
        weights = []
        parameters_list = []
        client_sizes = []
        quantum_params_this_round = []
        
        for client, fit_res in results:
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            parameters_list.append(parameters)
            num_examples = fit_res.num_examples
            weights.append(num_examples)
            client_sizes.append(num_examples)
            
            # 只收集真正的量子参数 'q_params'
            quantum_params = []
            for param_idx, param_name in enumerate(self.param_names):
                if param_name == 'q_params':  # 精确匹配
                    quantum_params.append(parameters[param_idx])
                    break  # 只有一个量子参数
            
            quantum_params_this_round.append(quantum_params)
        
        total_size = sum(weights)
        weights = [w / total_size for w in weights]
        
        # 记录量子参数用于周期性分析
        self.quantum_params_history.append(quantum_params_this_round)
        
        # 调试信息：添加在这里
        print(f"[DEBUG] Round {server_round}: Using {self.aggregation_method} aggregation")
        print(f"[DEBUG] Quantum params collected: {len(quantum_params_this_round)}")

        # 计算量子参数方差
        if quantum_params_this_round:
            quantum_variance = np.var([np.mean(np.abs(p[0])) for p in quantum_params_this_round if p])
            self.quantum_param_variance.append(quantum_variance)
        
        # 根据配置选择聚合方法
        if self.aggregation_method == "circular":
            aggregated_parameters = enhanced_circular_mean_aggregation(parameters_list, weights, self.param_names)
        else:  # arithmetic
            aggregated_parameters = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters[0] is not None:
                aggregated_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters[0])
        
        # 转换为Flower参数格式
        parameters = fl.common.ndarrays_to_parameters(aggregated_parameters)
        
        return parameters, {}
    
    def evaluate(self, server_round, parameters):
        """重写评估方法以记录历史数据"""
        if self.evaluate_fn is None:
            return None
        
        loss, metrics = self.evaluate_fn(server_round, parameters)
        accuracy = metrics.get("accuracy", 0.0)
        
        # 记录历史数据
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        
        # 每轮输出详细信息
        quantum_var = self.quantum_param_variance[-1] if self.quantum_param_variance else 0
        print(f"[ChallengeB] Round {server_round}: "
              f"Accuracy: {accuracy*100:.2f}%, "
              f"Loss: {loss:.4f}, "
              f"Quantum Param Variance: {quantum_var:.6f}, "
              f"Aggregation: {self.aggregation_method}")
        
        # 保存详细结果
        result_dir = FL_CONFIG["result_dir"]
        with open(f"{result_dir}/challengeB_{self.aggregation_method}_details.txt", "a") as f:
            f.write(f"Round {server_round}\tAccuracy: {accuracy*100:.4f}%\t"
                   f"Loss: {loss:.6f}\tQuantumVar: {quantum_var:.8f}\n")
        
        return loss, metrics
    
    def save_convergence_plot(self):
        """保存收敛曲线图"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.accuracy_history, 'b-', linewidth=2, marker='o')
        plt.title(f'Test Accuracy - {self.aggregation_method.capitalize()} Aggregation')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history, 'r-', linewidth=2, marker='o')
        plt.title(f'Test Loss - {self.aggregation_method.capitalize()} Aggregation')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{FL_CONFIG['result_dir']}/convergence_{self.aggregation_method}.pdf", bbox_inches='tight')
        plt.close()
        
        # 保存量子参数方差图
        if self.quantum_param_variance:
            plt.figure(figsize=(8, 4))
            plt.plot(self.quantum_param_variance, 'g-', linewidth=2, marker='o')
            plt.title(f'Quantum Parameter Variance - {self.aggregation_method.capitalize()} Aggregation')
            plt.xlabel('Communication Round')
            plt.ylabel('Variance')
            plt.grid(True)
            plt.savefig(f"{FL_CONFIG['result_dir']}/quantum_variance_{self.aggregation_method}.pdf", bbox_inches='tight')
            plt.close()

def run_challengeB_experiment():
    """运行挑战B对比实验 - 修复版本"""
    # 首先演示周期性冲突
    print("演示周期性参数冲突...")
    arithmetic_result, circular_result = demonstrate_period_mismatch()
    
    methods = ["arithmetic", "circular"]
    results = {}
    
    # 固定随机种子确保数据分布一致
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 预先生成共享的数据分布
    print("生成共享的数据分布...")
    FL_CONFIG["result_dir"] = "Result/exp_B/shared"
    os.makedirs(FL_CONFIG["result_dir"], exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train, x_test, y_test = load_data(classes=FL_CONFIG["classes"])
    client_datasets_shared = split_data_to_clients(
        x_train, y_train, 
        num_clients=FL_CONFIG["num_clients"],
        classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"],
        result_dir=FL_CONFIG["result_dir"]
    )
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running ChallengeB Experiment: {method.upper()} Aggregation")
        print(f"{'='*60}")
        
        # 更新配置
        FL_CONFIG["quantum_aggregation"] = method
        FL_CONFIG["result_dir"] = f"Result/exp_B/{method}"
        
        # 确保结果目录存在
        os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
        os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)
        
        # 使用共享的数据分布
        client_datasets = client_datasets_shared
        
        # 创建测试数据集
        test_dataset = CIFAR10Dataset(
            x_test, y_test, 
            transform=get_transform()
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=FL_CONFIG["batch_size"], 
            shuffle=False
        )
        
        # 初始化全局模型
        global_model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            input_channels=FL_CONFIG["input_channels"],
            device=device
        )
        
        # 准备初始参数和参数名称
        initial_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        param_names = list(global_model.state_dict().keys())
        initial_parameters = fl.common.ndarrays_to_parameters(initial_params)
        
        print(f"Model parameters: {param_names}")
        print(f"Quantum parameter identified: {'q_params' in param_names}")
        
        # 6. 定义服务器评估函数
        def evaluate_fn(server_round, parameters):
            return server_evaluate(
                server_round,
                parameters,
                global_model,
                test_loader,
                device,
                FL_CONFIG["result_dir"]
            )
        
        # 7. 创建专用策略
        strategy = ChallengeBStrategy(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            param_names=param_names,
            aggregation_method=method,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=FL_CONFIG["num_clients"],
            min_evaluate_clients=0,
            min_available_clients=FL_CONFIG["num_clients"],
            on_fit_config_fn=lambda server_round: {
                "server_round": server_round,
                "local_epochs": FL_CONFIG["local_epochs"],
                "learning_rate": FL_CONFIG["learning_rate"]
            }
        )
        
        # 8. 客户端创建函数
        def client_fn(cid: str):
            cid = int(cid)
            if cid not in client_datasets:
                raise ValueError(f"Invalid client ID: {cid}")
                
            model = HybridModelX(
                classical=FL_CONFIG["classical"],
                quantum=FL_CONFIG["quantum"],
                input_channels=FL_CONFIG["input_channels"],
                device=device
            )
            
            train_loader = DataLoader(
                client_datasets[cid], 
                batch_size=FL_CONFIG["batch_size"], 
                shuffle=True
            )
            
            return HybridClient(
                cid=cid,
                train_loader=train_loader,
                test_loader=test_loader,
                model=model,
                device=device,
                config=FL_CONFIG
            )
        
        # 9. 启动联邦学习
        print(f"Starting Federated Learning with {method} aggregation\n{'-'*50}")
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=FL_CONFIG["num_clients"],
            config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
            strategy=strategy,
            client_resources={
                "num_cpus": 6, 
                "num_gpus": 0.5 if torch.cuda.is_available() else 0
            }
        )
        
        # 10. 分析周期性行为
        periodicity_conflicts, round_variance = analyze_quantum_periodicity(
            strategy.quantum_params_history, method
        )
        
        # 11. 计算训练稳定性
        stability_metrics = calculate_training_stability(
            strategy.accuracy_history, 
            strategy.loss_history,
            strategy.quantum_param_variance
        )
        
        # 12. 保存结果和图表
        strategy.save_convergence_plot()
        results[method] = {
            "accuracy": strategy.accuracy_history,
            "loss": strategy.loss_history,
            "quantum_variance": strategy.quantum_param_variance,
            "periodicity_conflicts": periodicity_conflicts,
            "stability_metrics": stability_metrics,
            "round_variance": round_variance
        }
        
        print(f"\n{method.upper()} Aggregation Results:")
        print(f"  Final Accuracy: {strategy.accuracy_history[-1]*100:.2f}%")
        print(f"  Final Loss: {strategy.loss_history[-1]:.4f}")
        print(f"  Periodicity Conflicts: {len(periodicity_conflicts)}")
        print(f"  Stability Score: {stability_metrics['stability_score']:.4f}")
        print(f"  Accuracy STD: {stability_metrics['accuracy_std']:.4f}")
        print(f"  Loss Oscillations: {stability_metrics['loss_oscillations']}")
    
    # 生成对比图表
    generate_comparison_plots(results)

def generate_comparison_plots(results):
    """生成两种方法的对比图表"""
    plt.figure(figsize=(15, 10))
    
    # 准确率对比
    plt.subplot(2, 3, 1)
    for method, data in results.items():
        plt.plot(data["accuracy"], label=f'{method.capitalize()} Aggregation', 
                linewidth=2, marker='o')
    plt.title('Accuracy Comparison: Arithmetic vs Circular Aggregation')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 损失对比
    plt.subplot(2, 3, 2)
    for method, data in results.items():
        plt.plot(data["loss"], label=f'{method.capitalize()} Aggregation', 
                linewidth=2, marker='o')
    plt.title('Loss Comparison: Arithmetic vs Circular Aggregation')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 量子参数方差对比
    plt.subplot(2, 3, 3)
    for method, data in results.items():
        if data["quantum_variance"]:
            plt.plot(data["quantum_variance"], label=f'{method.capitalize()} Aggregation', 
                    linewidth=2, marker='o')
    plt.title('Quantum Parameter Variance Comparison')
    plt.xlabel('Communication Round')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # 周期性冲突对比
    plt.subplot(2, 3, 4)
    conflict_counts = [len(data["periodicity_conflicts"]) for data in results.values()]
    methods = list(results.keys())
    bars = plt.bar(methods, conflict_counts, color=['red', 'green'], alpha=0.7)
    plt.title('Periodicity Conflict Count Comparison')
    plt.ylabel('Number of Conflicts')
    for bar, count in zip(bars, conflict_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 稳定性评分对比
    plt.subplot(2, 3, 5)
    stability_scores = [data["stability_metrics"]["stability_score"] for data in results.values()]
    bars = plt.bar(methods, stability_scores, color=['red', 'green'], alpha=0.7)
    plt.title('Training Stability Score Comparison')
    plt.ylabel('Stability Score')
    for bar, score in zip(bars, stability_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score:.2f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 损失震荡次数对比
    plt.subplot(2, 3, 6)
    loss_oscillations = [data["stability_metrics"]["loss_oscillations"] for data in results.values()]
    bars = plt.bar(methods, loss_oscillations, color=['red', 'green'], alpha=0.7)
    plt.title('Loss Oscillation Count Comparison')
    plt.ylabel('Oscillation Count')
    for bar, count in zip(bars, loss_oscillations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("Result/exp_B/aggregation_comparison.pdf", bbox_inches='tight')
    plt.close()
    
    # 保存最终结果汇总
    with open("Result/exp_B/final_comparison.txt", "w") as f:
        f.write("Challenge B: Quantum Parameter Aggregation Methods Comparison\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Key Finding: Periodic Parameter Aggregation Challenge\n")
        f.write("Description: Quantum parameters are periodic (rotation angles).\n")
        f.write("Direct arithmetic averaging causes period mismatch and training instability.\n")
        f.write("Circular mean aggregation properly handles periodicity.\n\n")
        
        for method, data in results.items():
            final_acc = data["accuracy"][-1] * 100 if data["accuracy"] else 0
            final_loss = data["loss"][-1] if data["loss"] else 0
            avg_variance = np.mean(data["quantum_variance"]) if data["quantum_variance"] else 0
            conflict_count = len(data["periodicity_conflicts"])
            stability = data["stability_metrics"]["stability_score"]
            
            f.write(f"{method.upper()} Aggregation:\n")
            f.write(f"  Final Accuracy: {final_acc:.2f}%\n")
            f.write(f"  Final Loss: {final_loss:.4f}\n")
            f.write(f"  Average Quantum Variance: {avg_variance:.6f}\n")
            f.write(f"  Periodicity Conflicts: {conflict_count}\n")
            f.write(f"  Stability Score: {stability:.4f}\n")
            f.write(f"  Accuracy STD: {data['stability_metrics']['accuracy_std']:.4f}\n")
            f.write(f"  Loss Oscillations: {data['stability_metrics']['loss_oscillations']}\n\n")
        
        # 结论
        arithmetic_data = results.get("arithmetic", {})
        circular_data = results.get("circular", {})
        
        if arithmetic_data and circular_data:
            acc_improvement = circular_data["accuracy"][-1] - arithmetic_data["accuracy"][-1]
            stability_improvement = (circular_data["stability_metrics"]["stability_score"] - 
                                   arithmetic_data["stability_metrics"]["stability_score"])
            conflict_reduction = (len(arithmetic_data["periodicity_conflicts"]) - 
                               len(circular_data["periodicity_conflicts"]))
            
            f.write("CONCLUSION:\n")
            f.write(f"Circular mean aggregation provides:\n")
            f.write(f"  Accuracy Improvement: {acc_improvement*100:+.2f}%\n")
            f.write(f"  Stability Improvement: {stability_improvement:+.4f}\n")
            f.write(f"  Conflict Reduction: {conflict_reduction:+d} conflicts\n")
            
            if stability_improvement > 0 and conflict_reduction > 0:
                f.write("✅ SUCCESS: Challenge 2 (Quantum Parameter Periodicity) is CONFIRMED\n")
                f.write("   Arithmetic averaging causes period mismatch and instability.\n")
                f.write("   Circular mean properly handles quantum parameter periodicity.\n")
            else:
                f.write("⚠️  INCONCLUSIVE: Further investigation needed.\n")

if __name__ == "__main__":
    # 运行挑战B对比实验
    run_challengeB_experiment()