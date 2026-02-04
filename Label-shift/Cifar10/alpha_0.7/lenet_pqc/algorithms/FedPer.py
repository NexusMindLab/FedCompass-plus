import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_process.data_utils import load_data, split_data_to_clients, CIFAR10Dataset, get_transform
from models.modelx import HybridModelX
from federated.FedPer_FL import FedPerClient, FedPerStrategy, server_evaluate_fedper

# 配置参数
FL_CONFIG = {
    "server_round": 10,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "alpha": 0.7,
    "algorithm": "fedper",
    "classical": "lenet",   # 可选: resnet18 / lenet
    "quantum": "simple"     # 可选: qcnn / simple
}

FL_CONFIG["result_dir"] = f"lenet_pqc/result/FedPer"

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    x_train, y_train, x_test, y_test = load_data(
        classes=FL_CONFIG["classes"]
    )

    # 分配客户端数据
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_to_clients(
        x_train, y_train,
        num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"],
        classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"],
        result_dir=FL_CONFIG["result_dir"]
    )
    
    # 创建测试集
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=FL_CONFIG["batch_size"], shuffle=False)
    
    # 初始化模型
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
        device=device,
        input_channels=3
    )
    
    # 准备非个性化层初始参数
    def _is_personal_layer(layer_name):
        personal_keywords = ['quantum', 'fc']
        layer_name_lower = layer_name.lower()
        return any(keyword in layer_name_lower for keyword in personal_keywords)
    
    initial_params_base = [val.cpu().numpy() for name, val in global_model.state_dict().items() 
                          if not _is_personal_layer(name)]
    initial_parameters_base = fl.common.ndarrays_to_parameters(initial_params_base)
    
    # 定义评估函数
    def evaluate_fn_fedper(server_round, parameters):
        return server_evaluate_fedper(
            server_round,
            parameters,
            global_model,
            test_loader,
            device,
            FL_CONFIG["result_dir"]
        )
    
    # 创建FedPer策略
    strategy = FedPerStrategy(
        initial_parameters=initial_parameters_base,
        evaluate_fn=evaluate_fn_fedper,
        model_template=global_model,
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
    
    # 客户端创建函数
    def client_fn(cid: str):
        cid = int(cid)
        if cid not in CLIENT_TRAIN_LOADERS:
            raise ValueError(f"Invalid client ID: {cid}")

        model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            device=device,
        input_channels=3
    )

        return FedPerClient(
            cid=cid,
            train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG
        )
    
    # 启动联邦学习
    print(f"\nStarting Federated Learning with FedPer")
    print(f"Rounds: {FL_CONFIG['server_round']}, Clients: {FL_CONFIG['num_clients']}")
    print(f"Client config: local_epochs={FL_CONFIG['local_epochs']}, lr={FL_CONFIG['learning_rate']} (Personal layers kept local)")
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={"num_cpus": 0.5, "num_gpus": 0.1 if torch.cuda.is_available() else 0}
    )