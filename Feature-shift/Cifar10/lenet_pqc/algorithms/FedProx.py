import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_process.data_utils import load_data, split_data_feature_shift, CIFAR10Dataset, get_transform
from models.modelx import HybridModelX
from federated.FedProx_FL import HybridProxClient, FedProxHybrid, server_evaluate

# <--- 2. 定义你的大空间目录
MY_RAY_TEMP_DIR = "ray_temp"
os.makedirs(MY_RAY_TEMP_DIR, exist_ok=True)

# 配置参数
FL_CONFIG = {
    "server_round": 10,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "algorithm": "fedprox",
    "mu": 0.01,  # FedProx的近端项参数
    "result_dir": "result/FedProx",
    "classical": "lenet",   # 可选: resnet18 / lenet
    "quantum": "simple"     # 可选: qcnn / simple
}

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载全局数据
    x_train, y_train, x_test, y_test = load_data(
        classes=FL_CONFIG["classes"]
    )

    # 2. 为客户端分配数据 - 使用特征偏移
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_feature_shift(
        x_train,
        y_train,
        num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"],
        classes=FL_CONFIG["classes"],
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

    # 7. 创建FedProx策略
    strategy = FedProxHybrid(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        mu=FL_CONFIG["mu"],
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=FL_CONFIG["num_clients"],
        min_evaluate_clients=0,
        min_available_clients=FL_CONFIG["num_clients"],
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": FL_CONFIG["local_epochs"],
            "learning_rate": FL_CONFIG["learning_rate"],
            "mu": FL_CONFIG["mu"]
        }
    )

    # 8. 客户端创建函数
    def client_fn(cid: str):
        cid = int(cid)
        if cid not in CLIENT_TRAIN_LOADERS:
            raise ValueError(f"Invalid client ID: {cid}")

        # 创建客户端模型
        model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            device=device
        )

        return HybridProxClient(
            cid=cid,
            train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG
        )

    # 9. 启动联邦学习
    print(f"\nStarting Federated Learning with FedProx (Feature Shift)")
    print(f"Rounds: {FL_CONFIG['server_round']}, Clients: {FL_CONFIG['num_clients']}")
    print(f"Client config: local_epochs={FL_CONFIG['local_epochs']}, lr={FL_CONFIG['learning_rate']}, mu={FL_CONFIG['mu']}")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={
            "num_cpus": 6, 
            "num_gpus": 0.5 if torch.cuda.is_available() else 0
        },
        ray_init_args={
            "include_dashboard": False,
            "_temp_dir": MY_RAY_TEMP_DIR, 
            "num_cpus": 24, # 如果你之前没指定，可以加上
            "num_gpus": 2.0 # 如果你之前没指定，可以加上
        }
    )
