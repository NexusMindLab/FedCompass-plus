import os
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_process.data_utils import load_data, split_data_feature_shift, MNISTDataset, get_transform
from models.modelx import HybridModelX
from federated.FedBN_FL import FedBNClient, FedBNStrategy, server_evaluate_fedbn

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
    "result_dir": "result/FedBN",
    "classical": "resnet18",   # 可选: resnet18 / lenet
    "quantum": "qcnn",          # 可选: qcnn / simple
    "input_channels": 1        # 添加输入通道数配置，MNIST为1通道
}

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 输出模型配置
    print(f"\nModel Configuration:")
    print(f"  Classical feature extractor: {FL_CONFIG['classical']}")
    print(f"  Quantum layer: {FL_CONFIG['quantum']}")
    print(f"  Classes: {FL_CONFIG['classes']}")
    print(f"  Input channels: {FL_CONFIG['input_channels']}")
    print(f"  Clients: {FL_CONFIG['num_clients']}, Local epochs: {FL_CONFIG['local_epochs']}, LR: {FL_CONFIG['learning_rate']}\n")

    SEED = 42  # 可以是任意整数，但所有对比实验必须用同一个数字
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        
    # 加载数据
    x_train, y_train, x_test, y_test = load_data(
        classes=FL_CONFIG["classes"]
    )

    # 分配客户端数据 - 使用特征偏移
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_feature_shift(
        x_train, y_train,
        num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"],
        classes=FL_CONFIG["classes"],
        result_dir=FL_CONFIG["result_dir"]
    )

    # 3. 创建测试数据集
    test_dataset = MNISTDataset(  # 使用MNISTDataset
        x_test, 
        y_test, 
        transform=get_transform()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=FL_CONFIG["batch_size"], 
        shuffle=False
    )

    # 初始化模型
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
        device=device
    )

    # 准备非BatchNorm初始参数
    def _is_bn_layer(layer_name):
        bn_keywords = ['bn', 'batchnorm', 'batch_norm', 'norm']
        layer_name_lower = layer_name.lower()
        return any(keyword in layer_name_lower for keyword in bn_keywords)

    initial_params_no_bn = [val.cpu().numpy() for name, val in global_model.state_dict().items()
                           if not _is_bn_layer(name)]
    initial_parameters_no_bn = fl.common.ndarrays_to_parameters(initial_params_no_bn)

    # 定义评估函数
    def evaluate_fn_fedbn(server_round, parameters):
        return server_evaluate_fedbn(
            server_round,
            parameters,
            global_model,
            test_loader,
            device,
            FL_CONFIG["result_dir"]
        )

    # 创建FedBN策略
    strategy = FedBNStrategy(
        initial_parameters=initial_parameters_no_bn,
        evaluate_fn=evaluate_fn_fedbn,
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
            device=device
        )

        return FedBNClient(
            cid=cid,
            train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG
        )

    # 启动联邦学习
    print(f"\nStarting Federated Learning with FedBN (Feature Shift)")
    print(f"Rounds: {FL_CONFIG['server_round']}, Clients: {FL_CONFIG['num_clients']}")
    print(f"Client config: local_epochs={FL_CONFIG['local_epochs']}, lr={FL_CONFIG['learning_rate']} (BatchNorm layers kept local)")

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
