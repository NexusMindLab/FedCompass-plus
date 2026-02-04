# RQ.py (修改策略为FedNova)
import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_process.data_utils import load_data, split_data_to_clients, CIFAR10Dataset, get_transform
from models.modelx import HybridModelX
from federated.FedNova_FL import HybridClient, server_evaluate, FedNovaStrategy

# 配置参数
FL_CONFIG = {
    "server_round": 10,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "result_dir": "result/FedNova",
    "alpha": 0.7,              # Dirichlet分布参数0.1/0.3/0.7
    "classical": "lenet",   # 可选: resnet18 / lenet
    "quantum": "simple"          # 可选: qcnn / simple
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
    print(f"  Clients: {FL_CONFIG['num_clients']}, Local epochs: {FL_CONFIG['local_epochs']}, LR: {FL_CONFIG['learning_rate']}\n")

    SEED = 42  # 可以是任意整数，但所有对比实验必须用同一个数字
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        
    # 1. 加载全局数据
    x_train, y_train, x_test, y_test = load_data(
        classes=FL_CONFIG["classes"]
    )
    
    # 2. 分配客户端数据并获得分布信息
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_to_clients(
        x_train,
        y_train,
        num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"],
        classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"],
        result_dir=FL_CONFIG["result_dir"]
    )
    
    # 3. 创建测试数据集
    test_dataset = CIFAR10Dataset(
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
    # global_model = HybridModel()
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
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
    
    # 7. 创建FedNova策略
    strategy = FedNovaStrategy(
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
        if cid not in CLIENT_TRAIN_LOADERS:
            raise ValueError(f"Invalid client ID: {cid}")
            
        # 创建客户端模型
        # model = HybridModel()
        model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            device=device
        )
        
        # 创建客户端实例
        return HybridClient(
            cid=cid,
            train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG
        )
    
    # 9. 启动联邦学习
    print(f"Starting Federated Learning with FedNova\n{'-'*50}")
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