# FL.py
import os
import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# ----------------------------
# Evaluation helpers
# ----------------------------
def client_evaluation(model: nn.Module, test_loader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate a model on test_loader. Returns (loss, accuracy) where accuracy is fraction [0..1].
    """
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    loss /= len(test_loader) if len(test_loader) > 0 else 1.0
    accuracy = correct / total if total > 0 else 0.0

    return loss, accuracy


def server_evaluate(server_round: int, parameters, model: nn.Module, test_loader, device: torch.device, result_dir: str):
    """
    Set model parameters from Flower Parameters, evaluate on server test set, log results.
    Returns loss and dict of metrics (compat with Flower's evaluate_fn).
    """
    # Convert Parameters object -> list[np.ndarray]
    nds = parameters_to_ndarrays(parameters)

    # Map ndarrays into model.state_dict() order (keys order)
    sd_keys = list(model.state_dict().keys())
    if len(sd_keys) != len(nds):
        raise ValueError(f"Mismatch between model.state_dict() ({len(sd_keys)}) and provided parameters ({len(nds)})")

    # Build state_dict with consistent dtype (use float32 by default)
    state_dict = {}
    for k, v in zip(sd_keys, nds):
        state_dict[k] = torch.tensor(v, dtype=torch.float32)

    model.load_state_dict(state_dict, strict=True)

    loss, acc = client_evaluation(model, test_loader, device)

    print(f"\nServer round {server_round}: Evaluation loss {loss:.4f} / accuracy {acc*100:.4f} %")

    os.makedirs(f"{result_dir}/test", exist_ok=True)
    with open(f"{result_dir}/test/server_test.txt", "a") as f:
        f.write(f"Round {server_round}\tLoss: {loss:.4f}\tAccuracy: {acc*100:.4f}%\n")

    return loss, {"accuracy": acc}


# ----------------------------
# Helper: parameter (de)serialization consistent with state_dict order
# ----------------------------
def model_state_to_ndarrays(model: nn.Module) -> List[np.ndarray]:
    """
    Return model parameters as list of numpy arrays in the same order as state_dict().keys().
    """
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def load_ndarrays_to_model(model: nn.Module, ndarrays: List[np.ndarray]):
    """
    Given ndarrays in the same order as model.state_dict().keys(), load them into model.
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(ndarrays):
        raise ValueError("Length mismatch when loading ndarrays into model.")
    state_dict = {}
    for k, v in zip(keys, ndarrays):
        # default to float32
        state_dict[k] = torch.tensor(v, dtype=torch.float32)
    model.load_state_dict(state_dict, strict=True)


# ----------------------------
# SCAFFOLD: client-side training
# ----------------------------
def scaffold_client_training(
    model: nn.Module,
    train_loader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    cid: int,
    server_round: int,
    result_dir: str,
    c_local: List[torch.Tensor],
    c_global: List[torch.Tensor],
) -> Tuple[List[float], List[float], Dict[str, torch.Tensor], List[torch.Tensor]]:
    """
    Train model locally with SCAFFOLD correction.
    Returns (loss_history, acc_history, model_state_dict, c_local_new (list of tensors))
    - c_local and c_global should be lists of torch tensors (CPU or GPU). This function will move them to device.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Ensure c_local and c_global are tensors and move to device
    c_local = [torch.tensor(c, dtype=torch.float32).to(device) if not isinstance(c, torch.Tensor) else c.to(device) for c in c_local]
    c_global = [torch.tensor(c, dtype=torch.float32).to(device) if not isinstance(c, torch.Tensor) else c.to(device) for c in c_global]

    loss_history: List[float] = []
    acc_history: List[float] = []

    # Save initial parameters
    initial_params = [p.detach().clone() for p in model.parameters()]

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # SCAFFOLD gradient correction: p.grad += c_local - c_global
            with torch.no_grad():
                for p, c_l, c_g in zip(model.parameters(), c_local, c_global):
                    if p.grad is not None:
                        # ensure shapes match; if not, broadcast will error — that's good to catch
                        p.grad += (c_l - c_g)

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss /= len(train_loader) if len(train_loader) > 0 else 1.0
        epoch_acc = correct / total if total > 0 else 0.0
        epoch_acc_percent = epoch_acc * 100.0

        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc_percent)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = (
            f"[{current_time}] [SCAFFOLD Client {cid}, Round {server_round}] "
            f"Train Epoch: {epoch+1}/{epochs} \t Loss: {epoch_loss:.4f} | Acc: {epoch_acc_percent:.2f}%"
        )
        print(log_message, flush=True)

        os.makedirs(f"{result_dir}/train", exist_ok=True)
        with open(f"{result_dir}/train/client_{cid}_log.txt", "a") as f:
            f.write(f"[{current_time}] Round {server_round}\t Epoch {epoch + 1}\t Loss: {epoch_loss:.4f}\tAccuracy: {epoch_acc_percent:.4f}%\n")

    # Compute delta params and new c_local
    with torch.no_grad():
        final_params = [p.detach().clone() for p in model.parameters()]
        delta_params = [final_p - initial_p for final_p, initial_p in zip(final_params, initial_params)]

        # K = total local update steps. Protect from zero.
        num_batches = max(1, len(train_loader))
        K = max(1, epochs * num_batches)

        c_local_new: List[torch.Tensor] = []
        for c_l, c_g, delta_p in zip(c_local, c_global, delta_params):
            # c_local_new = c_l - c_g - delta_p / (K * lr)
            # ensure K*lr not zero
            denom = (K * learning_rate) if (K * learning_rate) != 0 else 1.0
            c_local_new.append(c_l - c_g - delta_p / denom)

    # Return model.state_dict() (on CPU) and c_local_new as tensors (on CPU)
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    c_local_new_cpu = [c.cpu() for c in c_local_new]

    return loss_history, acc_history, model_state, c_local_new_cpu


# ----------------------------
# SCAFFOLD: Flower client implementation
# ----------------------------
class ScaffoldClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementing SCAFFOLD corrections.
    """

    def __init__(self, cid: int, train_loader, test_loader, model: nn.Module, device: torch.device, config: Dict):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.config = config

        # SCAFFOLD specific variables
        self.c_local: Optional[List[torch.Tensor]] = None
        self.c_global: Optional[List[torch.Tensor]] = None
        self.server_round = 0

    # Keep parameter order consistent with model.state_dict().keys()
    def get_parameters(self, config):
        return model_state_to_ndarrays(self.model)

    def set_parameters(self, parameters):
        # parameters: list of ndarrays in same order as model.state_dict().keys()
        load_ndarrays_to_model(self.model, parameters)

    def fit(self, parameters, config):
        # config may contain server_round and c_global (ndarrays)
        self.server_round = config.get("server_round", 1)
        if "c_global" in config and config["c_global"] is not None:
            # convert list of ndarrays -> list of torch tensors (float32)
            self.c_global = [torch.tensor(arr, dtype=torch.float32) for arr in config["c_global"]]

        # Initialize c_local if first round
        if self.c_local is None:
            # Align shapes with model.parameters()
            self.c_local = [torch.zeros_like(p.detach()) for p in self.model.parameters()]
            if self.c_global is None:
                self.c_global = [torch.zeros_like(p.detach()) for p in self.model.parameters()]
            else:
                # If c_global provided but length mismatch, recreate
                if len(self.c_global) != len(list(self.model.parameters())):
                    self.c_global = [torch.zeros_like(p.detach()) for p in self.model.parameters()]

        # Load received global parameters into local model
        self.set_parameters(parameters)

        # Get training hyperparams (fall back to config stored in client)
        local_epochs = config.get("local_epochs", self.config.get("local_epochs", 1))
        learning_rate = config.get("learning_rate", self.config.get("learning_rate", 0.01))

        print(f"[SCAFFOLD Client {self.cid}] Starting local training for round {self.server_round}")

        loss_hist, acc_hist, model_state, c_local_new = scaffold_client_training(
            self.model,
            self.train_loader,
            local_epochs,
            learning_rate,
            self.device,
            self.cid,
            self.server_round,
            self.config.get("result_dir", "."),
            self.c_local,
            self.c_global
        )

        # delta_c = c_local_new - c_local
        delta_c = []
        for new, old in zip(c_local_new, self.c_local):
            delta_c.append((new.detach().cpu() - old.detach().cpu()).numpy())

        # update local control variable
        self.c_local = [c.clone().cpu() for c in c_local_new]

        # Update local model to trained state (model_state is cpu tensors)
        # Convert model_state back to model (careful with dtype)
        cpu_nds = [model_state[k].numpy() for k in model_state.keys()]
        load_ndarrays_to_model(self.model, cpu_nds)

        # Return updated parameters (ndarrays), num_examples, and metrics including delta_c
        return model_state_to_ndarrays(self.model), len(self.train_loader.dataset), {
            "loss": float(loss_hist[-1]) if len(loss_hist) > 0 else 0.0,
            "accuracy": float(acc_hist[-1]) if len(acc_hist) > 0 else 0.0,
            "delta_c": delta_c
        }

    def evaluate(self, parameters, config):
        # Load parameters
        self.set_parameters(parameters)
        loss, acc = client_evaluation(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}


# ----------------------------
# SCAFFOLD: Strategy implementation
# ----------------------------
class ScaffoldStrategy(fl.server.strategy.Strategy):
    """
    Flower server-side strategy implementing SCAFFOLD global control updates.
    """

    def __init__(self, initial_parameters, evaluate_fn, num_clients: int,
                 fraction_fit=1.0, fraction_evaluate=0.0,
                 min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # SCAFFOLD global control variable (list of torch tensors)
        self.c_global: Optional[List[torch.Tensor]] = None
        self.server_round = 0

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Convert Parameters -> ndarrays first (consistent)
        nds = parameters_to_ndarrays(parameters)
        if self.c_global is None:
            # initialize c_global zeros with same shapes as nds
            self.c_global = [torch.zeros_like(torch.tensor(nd, dtype=torch.float32)) for nd in nds]

        self.server_round = server_round

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {
            "server_round": server_round,
            "local_epochs": 5,
            "learning_rate": 0.001,
            # send global control as numpy arrays
            "c_global": [c.cpu().numpy() for c in self.c_global]
        }

        # IMPORTANT: return list of (client, FitIns), not (client, parameters, config)
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]


    def aggregate_fit(self, server_round: int, results, failures):
        """
        results: list of (client, FitRes)
        Each FitRes has .parameters and .num_examples and .metrics (where we expect "delta_c")
        """
        if not results:
            return None, {}

        # Build list of (ndarrays, num_examples) for parameter aggregation
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Collect delta_c lists and corresponding num_examples
        delta_c_list = []
        num_examples_for_delta = []
        for _, fit_res in results:
            if hasattr(fit_res, "metrics") and fit_res.metrics is not None and "delta_c" in fit_res.metrics:
                delta_c = fit_res.metrics["delta_c"]
                if delta_c is not None:
                    delta_c_list.append(delta_c)
                    num_examples_for_delta.append(fit_res.num_examples)

        # Update global control variable c_global if deltas provided
        if delta_c_list:
            # Weighted average of delta_c by num_examples (protect from zero-sum)
            total_n = sum(num_examples_for_delta) if sum(num_examples_for_delta) > 0 else len(num_examples_for_delta)
            avg_delta_c = [np.zeros_like(delta_c_list[0][i]) for i in range(len(delta_c_list[0]))]
            for delta_c, n in zip(delta_c_list, num_examples_for_delta):
                for i, dc in enumerate(delta_c):
                    avg_delta_c[i] += dc * (n / total_n)
            # Update c_global (ensure same length)
            if len(self.c_global) != len(avg_delta_c):
                # Re-init if shape mismatch (defensive)
                self.c_global = [torch.zeros_like(torch.tensor(arr, dtype=torch.float32)) for arr in avg_delta_c]
            for i in range(len(self.c_global)):
                self.c_global[i] = self.c_global[i] + torch.tensor(avg_delta_c[i], dtype=torch.float32)

        # Aggregate model parameters with weighted average by num_examples
        aggregated_ndarrays = [
            np.average([w[i] for w, _ in weights_results], axis=0, weights=[n for _, n in weights_results])
            for i in range(len(weights_results[0][0]))
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        return parameters_aggregated, {}

    
    def configure_evaluate(self, server_round: int, parameters, client_manager):
        sample_size, min_num_clients = self.num_evaluate_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Return list of (client, EvaluateIns)
        return [(client, fl.common.EvaluateIns(parameters, {})) for client in clients]

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        loss_aggregated = np.average(
            [evaluate_res.loss for _, evaluate_res in results],
            weights=[evaluate_res.num_examples for _, evaluate_res in results],
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters)

    def num_fit_clients(self, num_available_clients: int):
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients

    def num_evaluate_clients(self, num_available_clients: int):
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients



