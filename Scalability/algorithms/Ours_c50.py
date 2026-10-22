import os
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
from data_process.data_utils import load_data, split_data_to_clients, CIFAR10Dataset, get_transform
from models.modelx import HybridModelX
from federated.our_FL import HybridClient, FedAvgHybridClustered, server_evaluate
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

MY_RAY_TEMP_DIR = os.environ.get("MY_RAY_TEMP_DIR", "/tmp/ray_temp")
os.makedirs(MY_RAY_TEMP_DIR, exist_ok=True)

FL_CONFIG = {
    "server_round": 50,
    "local_epochs": 5,
    "num_clients": 50,
    "learning_rate": 0.001,
    "batch_size": 32,
    "classes": [0, 1, 2, 3],
    "result_dir": "result/Ours_c50",
    "alpha": 0.3,
    "classical": "resnet18",
    "quantum": "qcnn"
}

if __name__ == "__main__":
    os.makedirs(f"{FL_CONFIG['result_dir']}/train", exist_ok=True)
    os.makedirs(f"{FL_CONFIG['result_dir']}/test", exist_ok=True)
    LOCAL_LOG_DIR = os.path.join(MY_RAY_TEMP_DIR, "log_" + FL_CONFIG["result_dir"].split("/")[-1])
    os.makedirs(f"{LOCAL_LOG_DIR}/train", exist_ok=True)
    os.makedirs(f"{LOCAL_LOG_DIR}/test", exist_ok=True)
    FL_CONFIG["local_log_dir"] = LOCAL_LOG_DIR
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\n=== Ours | clients={FL_CONFIG['num_clients']} | rounds={FL_CONFIG['server_round']} ===\n")

    SEED = 42
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

    x_train, y_train, x_test, y_test = load_data(classes=FL_CONFIG["classes"])
    CLIENT_TRAIN_LOADERS, CLIENT_CLASS_DISTS = split_data_to_clients(
        x_train, y_train, num_clients=FL_CONFIG["num_clients"],
        batch_size=FL_CONFIG["batch_size"], classes=FL_CONFIG["classes"],
        alpha=FL_CONFIG["alpha"], result_dir=FL_CONFIG["result_dir"])
    test_dataset = CIFAR10Dataset(x_test, y_test, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=FL_CONFIG["batch_size"], shuffle=False)

    global_model = HybridModelX(classical=FL_CONFIG["classical"], quantum=FL_CONFIG["quantum"], device=device)
    initial_params = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params)
    param_keys = list(global_model.state_dict().keys())

    def evaluate_fn(server_round, parameters):
        return server_evaluate(server_round, parameters, global_model, test_loader, device, FL_CONFIG["result_dir"])

    num_clusters = min(3, FL_CONFIG["num_clients"])
    strategy = FedAvgHybridClustered(
        initial_parameters=initial_parameters, evaluate_fn=evaluate_fn,
        param_keys=param_keys, num_clusters=num_clusters, qparam_name='q_params',
        lr_q=0.002, fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=FL_CONFIG["num_clients"], min_evaluate_clients=0,
        min_available_clients=FL_CONFIG["num_clients"],
        on_fit_config_fn=lambda sr: {"server_round": sr, "local_epochs": FL_CONFIG["local_epochs"], "learning_rate": FL_CONFIG["learning_rate"]})

    def client_fn(cid: str):
        cid = int(cid)
        if cid not in CLIENT_TRAIN_LOADERS: raise ValueError(f"Invalid client ID: {cid}")
        model = HybridModelX(classical=FL_CONFIG["classical"], quantum=FL_CONFIG["quantum"], device=device)
        return HybridClient(cid=cid, train_loader=CLIENT_TRAIN_LOADERS[cid],
                           test_loader=test_loader, model=model, device=device,
                           config=FL_CONFIG, class_counts=CLIENT_CLASS_DISTS[cid])

    print(f"Starting {FL_CONFIG['result_dir']}")
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=FL_CONFIG["num_clients"],
        config=fl.server.ServerConfig(num_rounds=FL_CONFIG["server_round"]), strategy=strategy,
        client_resources={"num_cpus": int(os.environ.get("RAY_CLIENT_NUM_CPUS", "4")),
                          "num_gpus": float(os.environ.get("RAY_CLIENT_NUM_GPUS", "0.5")) if torch.cuda.is_available() else 0},
        ray_init_args={"include_dashboard": False, "_temp_dir": MY_RAY_TEMP_DIR,
                        "num_cpus": int(os.environ.get("RAY_NUM_CPUS", "24")),
                        "num_gpus": float(os.environ.get("RAY_NUM_GPUS", "2.0"))})

    print(f"\nCopying training logs from local temp to result directory...")
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
