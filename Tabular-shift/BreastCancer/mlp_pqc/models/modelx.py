# modelx.py — Breast Cancer Tabular Model
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import math
from typing import Tuple

# =====================================================
# 量子层 (与图像实验保持一致)
# =====================================================
def U_SU4(params, wires):
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

def Pooling_ansatz1(params, wires):
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])

def QCNN_structure(params):
    """QCNN structure for 8 qubits."""
    param1, param2, param3 = params[:15], params[15:30], params[30:45]
    param4, param5, param6 = params[45:47], params[47:49], params[49:51]

    U_SU4(param1, wires=[0, 7])
    for i in range(0, 8, 2):
        U_SU4(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U_SU4(param1, wires=[i, i + 1])
    for i in range(0, 8, 2):
        Pooling_ansatz1(param4, wires=[i + 1, i])

    U_SU4(param2, wires=[0, 6])
    U_SU4(param2, wires=[0, 2])
    U_SU4(param2, wires=[4, 6])
    U_SU4(param2, wires=[2, 4])
    Pooling_ansatz1(param5, wires=[2, 0])
    Pooling_ansatz1(param5, wires=[6, 4])

    U_SU4(param3, wires=[0, 4])
    Pooling_ansatz1(param6, wires=[0, 4])

# =====================================================
# 经典层: MLP (表格数据特征提取器)
# =====================================================
def build_mlp_feature_extractor(out_dim=8, input_dim=30):
    """MLP 特征提取器，适用于表格数据。
    input_dim=30 (Breast Cancer 特征数)
    out_dim=8 (简单量子) 或 16 (QCNN)
    """
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Dropout(0.3),
        nn.Linear(32, out_dim),
    )

# =====================================================
# 混合模型
# =====================================================
class HybridModelX(nn.Module):
    def __init__(self, classical="mlp", quantum="qcnn",
                 n_qubits=8, n_layers=4, n_classes=2, device=None, input_dim=30):
        super().__init__()
        self.classical = classical
        self.quantum = quantum
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim

        if quantum == "qcnn":
            classical_out_dim = 2 * self.n_qubits  # 16
        else:
            classical_out_dim = self.n_qubits       # 8

        # MLP 特征提取器
        self.feature_extractor = build_mlp_feature_extractor(
            out_dim=classical_out_dim, input_dim=self.input_dim
        ).to(self.device)

        # --------- 量子部分 ---------
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        if quantum == "qcnn":
            self.quantum_params_size = 51
            self.q_params = nn.Parameter(
                torch.randn(self.quantum_params_size, device=self.device) * 0.01
            )

            @qml.qnode(self.dev, interface="torch")
            def qcnn_circuit(features, q_params):
                for i in range(min(n_qubits, len(features))):
                    qml.RX(features[i], wires=i)
                    if i + n_qubits < len(features):
                        qml.RY(features[i + n_qubits], wires=i)
                QCNN_structure(q_params)
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]

            self.quantum_circuit = qcnn_circuit

        elif quantum == "simple":
            self.quantum_params_size = self.n_layers * self.n_qubits * 3
            self.q_params = nn.Parameter(
                torch.randn(self.quantum_params_size, device=self.device) * 0.1
            )

            @qml.qnode(self.dev, interface="torch")
            def simple_circuit(features, q_params):
                qparams = q_params.view(self.n_layers, self.n_qubits, 3)
                for i in range(self.n_qubits):
                    if i < len(features):
                        qml.RY(features[i], wires=i)
                for layer in range(self.n_layers):
                    for qubit in range(self.n_qubits):
                        qml.RY(qparams[layer, qubit, 0], wires=qubit)
                        qml.RZ(qparams[layer, qubit, 1], wires=qubit)
                        qml.RY(qparams[layer, qubit, 2], wires=qubit)
                    for qubit in range(self.n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                    if self.n_qubits > 2:
                        qml.CNOT(wires=[self.n_qubits - 1, 0])
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]

            self.quantum_circuit = simple_circuit

        else:
            raise ValueError(f"Unsupported quantum model: {quantum}")

    def forward(self, x):
        x = x.to(self.device)
        features = self.feature_extractor(x)
        features = torch.tanh(features)
        features = (features + 1) * math.pi
        outputs = []
        for i in range(features.size(0)):
            out = self.quantum_circuit(features[i], self.q_params)
            outputs.append(torch.stack(out))
        return torch.stack(outputs).float()

    def get_quantum_params(self):
        return self.q_params.detach().cpu().numpy()

    def set_quantum_params(self, params):
        self.q_params.data = torch.tensor(params, device=self.device, dtype=torch.float32)

    def get_classical_params(self):
        return list(self.feature_extractor.parameters())

    def get_quantum_params_list(self):
        return [self.q_params]

    def get_all_params(self) -> Tuple[list, list]:
        classical_params = []
        quantum_params = []
        for name, param in self.named_parameters():
            if 'feature_extractor' in name:
                classical_params.append(param)
            elif 'q_params' in name:
                quantum_params.append(param)
        return classical_params, quantum_params
