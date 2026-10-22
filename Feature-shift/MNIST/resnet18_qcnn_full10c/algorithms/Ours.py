import os
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
# 导入其他模块
from data_process.data_utils import load_data, split_data_feature_shift, MNISTDataset, get_transform
from models.modelx import HybridModelX
from federated.our_FL import HybridClient, FedAvgHybridClustered, server_evaluate
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

# <--- 2. 定义你的大空间目录
# 建议写死或者从环境变量读，但为了保险，我们直接写死路径
MY_RAY_TEMP_DIR = os.environ.get("MY_RAY_TEMP_DIR", "/tmp/ray_temp")
os.makedirs(MY_RAY_TEMP_DIR, exist_ok=True)

FL_CONFIG = {
    "server_round": 10,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0,1,2,3,4,5,6,7,8,9],
    "result_dir": "result/Ours_full10c", # 适配0.1参数
    "classical": "resnet18",   # 可选: resnet18 / lenet
    "quantum": "qcnn",          # 可选: qcnn / simple
    "input_channels": 1        # 添加输入通道数配置，MNIST为1通道
}

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)

    # 创建本地临时日志目录（避免NFS并发I/O错误）
    LOCAL_LOG_DIR = os.path.join(MY_RAY_TEMP_DIR, "log_" + FL_CONFIG["result_dir"].split("/")[-1])
    os.makedirs(f"{LOCAL_LOG_DIR}/train", exist_ok=True)
    os.makedirs(f"{LOCAL_LOG_DIR}/test", exist_ok=True)
    FL_CONFIG["local_log_dir"] = LOCAL_LOG_DIR

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

    SEED = 42  # 可以是任意整数，但所有对比实验必须用同一个数字
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # 1. 加载数据
    x_train, y_train, x_test, y_test = load_data(classes=FL_CONFIG["classes"])

    # 2. 分配客户端数据并获得分布信息
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_feature_shift(
        x_train,
        y_train,
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

    # 4. 初始化全局模型
    global_model = HybridModelX(
        classical=FL_CONFIG["classical"],
        quantum=FL_CONFIG["quantum"],
        input_channels=FL_CONFIG["input_channels"],  # 添加输入通道数
        n_classes=10,
        n_qubits=10,
        device=device
    )

    # 5. 初始参数与 param_keys
    initial_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params)
    param_keys = list(global_model.state_dict().keys())

    # 6. 定义服务器评估函数
    def evaluate_fn(server_round, parameters):
        return server_evaluate(server_round, parameters, global_model, test_loader, device, FL_CONFIG["result_dir"])

    # 7. 策略：使用簇数为 min(3, num_clients)
    num_clusters = min(3, FL_CONFIG["num_clients"])
    strategy = FedAvgHybridClustered(
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        param_keys=param_keys,
        num_clusters=num_clusters,
        qparam_name='q_params',
        lr_q=0.002,
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

    # 8. 客户端创建函数，注入 class distribution
    def client_fn(cid: str):
        cid = int(cid)
        if cid not in CLIENT_TRAIN_LOADERS:
            raise ValueError(f"Invalid client ID: {cid}")
        
        model = HybridModelX(
            classical=FL_CONFIG["classical"],
            quantum=FL_CONFIG["quantum"],
            input_channels=FL_CONFIG["input_channels"],  # 添加输入通道数
            n_classes=10,
            n_qubits=10,
            device=device
        )
        return HybridClient(
            cid=cid,
            train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader,
            model=model,
            device=device,
            config=FL_CONFIG,
            class_counts=CLIENT_CLASS_DISTS[cid]
        )

    # 9. 启动仿真
    print(f"Starting Federated Learning\n{'-'*50}")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={"num_cpus": int(os.environ.get("RAY_CLIENT_NUM_CPUS", "4")), "num_gpus": float(os.environ.get("RAY_CLIENT_NUM_GPUS", "0.5")) if torch.cuda.is_available() else 0},
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