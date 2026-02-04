import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_utils import load_data, split_data_to_clients, FashionMNISTDataset, get_transform  # 修改导入
from modely import HybridModelX
from FL import HybridClient, FedAvgHybrid, server_evaluate

# 配置参数
FL_CONFIG = {
    "server_round": 2,
    "local_epochs": 2,
    "num_clients": 3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "result_dir": "Result/0.7",  
    "alpha": 0.7,              # Dirichlet分布参数0.1\0.3\0.7
    "classical": "resnet18",   # 可选: resnet18 / lenet
    "quantum": "qcnn",          # 可选: qcnn / simple
    "input_channels": 1        # 添加输入通道数配置，FashionMNIST为1通道
}

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 输出模型配置
    print(f"\nModel Configuration:")
    print(f"  Classical feature extractor: {FL_CONFIG['classical']}")
    print(f"  Quantum layer: {FL_CONFIG['quantum']}")
    print(f"  Classes: {FL_CONFIG['classes']}")
    print(f"  Input channels: {FL_CONFIG['input_channels']}")
    print(f"  Clients: {FL_CONFIG['num_clients']}, Local epochs: {FL_CONFIG['local_epochs']}, LR: {FL_CONFIG['learning_rate']}\n")

    # 1. 加载全局数据
    x_train, y_train, x_test, y_test = load_data(
        classes=FL_CONFIG["classes"]
    )
    # 测试代码
    x_train = x_train[:200]
    y_train = y_train[:200]
    x_test = x_test[:min(200, len(x_test))]
    y_test = y_test[:min(200, len(y_test))]

    # 2. 为客户端分配数据
    client_datasets = split_data_to_clients(
        x_train, 
        y_train, 
        num_clients=FL_CONFIG["num_clients"],
        classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"],
        result_dir=FL_CONFIG["result_dir"]
    )
    
    # 3. 创建测试数据集
    test_dataset = FashionMNISTDataset(  # 使用FashionMNISTDataset
        x_test, 
        y_test, 
        transform=get_transform()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=FL_CONFIG["batch_size"], 
        shuffle=False
    )
    
    # 4. 初始化全局模型
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
        input_channels=FL_CONFIG["input_channels"],  # 添加输入通道数
        device=device
    )

    
    # 5. 准备初始参数
    initial_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_params)
    
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
    
    # 7. 创建联邦学习策略
    strategy = FedAvgHybrid(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
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
            
        # 创建客户端模型
        model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            input_channels=FL_CONFIG["input_channels"],  # 添加输入通道数
            device=device
        )
        
        # 创建客户端数据加载器
        train_loader = DataLoader(
            client_datasets[cid], 
            batch_size=FL_CONFIG["batch_size"], 
            shuffle=True
        )
        
        # 创建客户端实例
        return HybridClient(
            cid=cid,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG
        )
    
    # 9. 启动联邦学习
    print(f"Starting Federated Learning\n{'-'*50}")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={
            "num_cpus": 4, 
            "num_gpus": 0.2 if torch.cuda.is_available() else 0
        }
    )