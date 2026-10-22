import os
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_process.data_utils import load_data, split_data_to_clients, TabularDataset, get_transform
from models.modelx import HybridModelX
from federated.FedBN_FL import FedBNClient, FedBNStrategy, server_evaluate_fedbn
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

MY_RAY_TEMP_DIR = os.environ.get("MY_RAY_TEMP_DIR", "/tmp/ray_temp")
os.makedirs(MY_RAY_TEMP_DIR, exist_ok=True)

# 结果目录锚定到脚本所在位置（模型根目录/result），不依赖运行时 cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FL_CONFIG = {
    "server_round": 20,
    "local_epochs": 5,
    "num_clients": 10,
    "learning_rate": 0.001,
    "batch_size": 16,
    "classes": [0, 1],
    "result_dir": os.path.join(BASE_DIR, "..", "result", "FedBN_BC_a0.3"),
    "alpha": 0.3,
    "classical": "mlp",
    "quantum": "simple",
    "input_dim": 30
}

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)

    # 创建本地临时日志目录（避免NFS并发I/O错误）
    LOCAL_LOG_DIR = os.path.join(MY_RAY_TEMP_DIR, "log_" + os.path.basename(FL_CONFIG["result_dir"]))
    os.makedirs(f"{LOCAL_LOG_DIR}/train", exist_ok=True)
    os.makedirs(f"{LOCAL_LOG_DIR}/test", exist_ok=True)
    FL_CONFIG["local_log_dir"] = LOCAL_LOG_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nModel: MLP + simple | Dataset: Breast Cancer | Alpha: 0.3")

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    x_train, y_train, x_test, y_test = load_data(classes=FL_CONFIG["classes"])

    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_to_clients(
        x_train, y_train, num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"], classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"], result_dir=FL_CONFIG["result_dir"]
    )

    test_dataset = TabularDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=FL_CONFIG["batch_size"], shuffle=False)

    global_model = HybridModelX(
        classical=FL_CONFIG["classical"], quantum=FL_CONFIG["quantum"],
        n_classes=2, n_qubits=8, input_dim=FL_CONFIG["input_dim"], device=device
    )

    initial_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params)
    param_keys = list(global_model.state_dict().keys())

    def evaluate_fn(server_round, parameters):
        return server_evaluate_fedbn(server_round, parameters, global_model, test_loader, device, FL_CONFIG["result_dir"])

    def _is_bn_layer(layer_name):
        bn_keywords = ['bn', 'batchnorm', 'batch_norm', 'norm']
        return any(k in layer_name.lower() for k in bn_keywords)
    initial_params_no_bn = [val.cpu().numpy() for name, val in global_model.state_dict().items()
                           if not _is_bn_layer(name)]
    initial_parameters_no_bn = fl.common.ndarrays_to_parameters(initial_params_no_bn)
    def evaluate_fn_fedbn(server_round, parameters):
        return server_evaluate_fedbn(server_round, parameters, global_model, test_loader, device, FL_CONFIG["result_dir"])
    strategy = FedBNStrategy(
        initial_parameters=initial_parameters_no_bn, evaluate_fn=evaluate_fn_fedbn,
        model_template=global_model, fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=FL_CONFIG["num_clients"], min_evaluate_clients=0,
        min_available_clients=FL_CONFIG["num_clients"],
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round, "local_epochs": FL_CONFIG["local_epochs"],
            "learning_rate": FL_CONFIG["learning_rate"]
        }
    )


    def client_fn(cid: str):
        cid = int(cid)
        if cid not in CLIENT_TRAIN_LOADERS:
            raise ValueError(f"Invalid client ID: {cid}")
        model = HybridModelX(
            classical=FL_CONFIG["classical"], quantum=FL_CONFIG["quantum"],
            n_classes=2, n_qubits=8, input_dim=FL_CONFIG["input_dim"], device=device
        )
        return FedBNClient(
            cid=cid, train_loader=CLIENT_TRAIN_LOADERS[cid],
            test_loader=test_loader, model=model, device=device,
            config=FL_CONFIG
        )

    print(f"Starting Federated Learning (Breast Cancer, FedBN (BC, alpha=0.3))")
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]),
        strategy=strategy,
        client_resources={"num_cpus": int(os.environ.get("RAY_CLIENT_NUM_CPUS", "4")), "num_gpus": float(os.environ.get("RAY_CLIENT_NUM_GPUS", "0.5")) if torch.cuda.is_available() else 0},
        ray_init_args={"include_dashboard": False, "_temp_dir": MY_RAY_TEMP_DIR,
                       "num_cpus": int(os.environ.get("RAY_NUM_CPUS", "24")),
                       "num_gpus": float(os.environ.get("RAY_NUM_GPUS", "2.0"))}
    )

    # ==== 将本地临时日志复制回NFS结果目录（训练结束后执行） ====
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

