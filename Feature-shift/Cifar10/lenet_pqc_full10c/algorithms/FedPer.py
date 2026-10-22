import os
import shutil
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data_process.data_utils import load_data, split_data_feature_shift, CIFAR10Dataset, get_transform
from models.modelx import HybridModelX
from federated.FedPer_FL import FedPerClient, FedPerStrategy, server_evaluate_fedper

# <--- 2. 定义你的大空间目录
# 建议写死或者从环境变量读，但为了保险，我们直接写死路径
MY_RAY_TEMP_DIR = os.environ.get("MY_RAY_TEMP_DIR", "/tmp/ray_temp")
os.makedirs(MY_RAY_TEMP_DIR, exist_ok=True)

# 配置参数
FL_CONFIG = {
    "server_round": 10,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0,1,2,3,4,5,6,7,8,9],
    "algorithm": "fedper",
    "result_dir": "result/FedPer_full10c",
    "classical": "lenet",   # 可选: resnet18 / lenet
    "quantum": "simple"     # 可选: qcnn / simple
}

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)

    # 创建本地临时日志目录（避免NFS并发I/O错误）
    LOCAL_LOG_DIR = os.path.join(MY_RAY_TEMP_DIR, "log_" + FL_CONFIG["result_dir"].split("/")[-1])
    os.makedirs(f"{LOCAL_LOG_DIR}/train", exist_ok=True)
    os.makedirs(f"{LOCAL_LOG_DIR}/test", exist_ok=True)
    FL_CONFIG["local_log_dir"] = LOCAL_LOG_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # 创建测试集
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=FL_CONFIG["batch_size"], shuffle=False)

    # 初始化模型
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
        n_classes=10,
        n_qubits=10,
        device=device
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
            n_classes=10,
            n_qubits=10,
            device=device
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
    print(f"\nStarting Federated Learning with FedPer (Feature Shift)")
    print(f"Rounds: {FL_CONFIG['server_round']}, Clients: {FL_CONFIG['num_clients']}")
    print(f"Client config: local_epochs={FL_CONFIG['local_epochs']}, lr={FL_CONFIG['learning_rate']} (Personal layers kept local)")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={
            "num_cpus": int(os.environ.get("RAY_CLIENT_NUM_CPUS", "4")), 
            "num_gpus": float(os.environ.get("RAY_CLIENT_NUM_GPUS", "0.5")) if torch.cuda.is_available() else 0
        },
        ray_init_args={
            "include_dashboard": False,
            "_temp_dir": MY_RAY_TEMP_DIR, 
            "num_cpus": int(os.environ.get("RAY_NUM_CPUS", "24")), # 如果你之前没指定，可以加上
            "num_gpus": float(os.environ.get("RAY_NUM_GPUS", "2.0")) # 如果你之前没指定，可以加上
        }
    )

    # ==== 将本地临时日志复制回NFS结果目录 ====
    print(f"\nCopying training logs from local temp to result directory...")
    print(f"  Source: {LOCAL_LOG_DIR}")
    print(f"  Target: {FL_CONFIG['result_dir']}")
    for subdir in ['train', 'test']:
        src_dir = os.path.join(LOCAL_LOG_DIR, subdir)
        dst_dir = os.path.join(FL_CONFIG['result_dir'], subdir)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.isdir(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"  Copied: {subdir}/{filename}")
    print("Logs copy completed.\n")


    # ==== 将本地临时日志复制回NFS结果目录 ====
    print(f"\nCopying training logs from local temp to result directory...")
    print(f"  Source: {LOCAL_LOG_DIR}")
    print(f"  Target: {FL_CONFIG['result_dir']}")
    for subdir in ['train', 'test']:
        src_dir = os.path.join(LOCAL_LOG_DIR, subdir)
        dst_dir = os.path.join(FL_CONFIG['result_dir'], subdir)
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.isdir(src_dir):
            for filename in os.listdir(src_dir):
                src_file = os.path.join(src_dir, filename)
                dst_file = os.path.join(dst_dir, filename)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"  Copied: {subdir}/{filename}")
    print("Logs copy completed.\n")
