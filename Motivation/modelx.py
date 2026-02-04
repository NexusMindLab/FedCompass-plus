# modelx.py
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
from torchvision import models
import math

# =====================================================
# 量子层
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
# 经典层定义
# =====================================================
def build_resnet18_feature_extractor(out_dim=16, input_channels=3):
    base_model = models.resnet18(weights=None)

    # 修改第一层以适应不同的输入通道数
    if input_channels != 3:
        base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    layers = nn.Sequential(
        base_model.conv1,
        base_model.bn1,
        base_model.relu,
        base_model.maxpool,
        base_model.layer1,
        base_model.layer2,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, out_dim),
    )
    return layers

def build_lenet_feature_extractor(out_dim=8, input_channels=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, 6, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16*6*6, 64),  # 改为576
        nn.ReLU(),
        nn.Linear(64, out_dim)
    )

# =====================================================
# 模块化混合模型
# =====================================================
class HybridModelX(nn.Module):
    def __init__(self, classical="resnet18", quantum="qcnn",
                 n_qubits=8, n_layers=4, n_classes=4, device=None, input_channels=3):
        super().__init__()
        self.classical = classical
        self.quantum = quantum
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channels = input_channels
        
        # 确定经典特征提取器的输出维度
        if quantum == "qcnn":
            # QCNN需要2*n_qubits维特征进行编码
            classical_out_dim = 2 * self.n_qubits
        else:
            # 简单量子电路只需要n_qubits维特征
            classical_out_dim = self.n_qubits
            
        # --------- 经典特征提取器 ---------
        if classical == "resnet18":
            self.feature_extractor = build_resnet18_feature_extractor(
                out_dim=classical_out_dim, 
                input_channels=self.input_channels
            )
        elif classical == "lenet":
            self.feature_extractor = build_lenet_feature_extractor(
                out_dim=classical_out_dim, 
                input_channels=self.input_channels
            )
        else:
            raise ValueError("Unsupported classical model")
        
        # 将特征提取器移动到设备
        self.feature_extractor = self.feature_extractor.to(self.device)

        # --------- 量子部分 ---------
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        if quantum == "qcnn":
            self.quantum_params_size = 51
            self.q_params = nn.Parameter(
                torch.randn(self.quantum_params_size, device=self.device) * 0.01
            )

            @qml.qnode(self.dev, interface="torch")
            def qcnn_circuit(features, q_params):
                # 量子特征映射 - 使用所有特征
                for i in range(min(n_qubits, len(features))):
                    qml.RX(features[i], wires=i)
                    if i + n_qubits < len(features):
                        qml.RY(features[i + n_qubits], wires=i)

                # 应用QCNN结构
                QCNN_structure(q_params)
                # 测量
                return [qml.expval(qml.PauliZ(i)) for i in range(self.n_classes)]
            
            self.quantum_circuit = qcnn_circuit

        elif quantum == "simple":
            # 量子权重参数
            self.quantum_params_size = self.n_layers * self.n_qubits * 3
            self.q_params = nn.Parameter(
                torch.randn(self.quantum_params_size, device=self.device) * 0.1
            )

            @qml.qnode(self.dev, interface="torch")
            def simple_circuit(features, q_params):
                qparams = q_params.view(self.n_layers, self.n_qubits, 3)

                # 数据编码
                for i in range(self.n_qubits):
                    if i < len(features):
                        qml.RY(features[i], wires=i)

                # 多层参数化 + 纠缠
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
        # 确保输入在正确的设备上
        x = x.to(self.device)
        
        # 经典特征提取
        if self.classical == "lenet" and x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度
            
        features = self.feature_extractor(x)
        
        # 特征归一化到 [0, 2π]
        features = torch.tanh(features)
        features = (features + 1) * math.pi
        
        # 处理量子电路
        outputs = []
        for i in range(features.size(0)):
            # 执行量子电路
            out = self.quantum_circuit(features[i], self.q_params)
            outputs.append(torch.stack(out))
            
        return torch.stack(outputs).float()

    def get_quantum_params(self):
        """返回量子参数"""
        return self.q_params.detach().cpu().numpy()
    
    def set_quantum_params(self, params):
        """设置量子参数"""
        self.q_params.data = torch.tensor(params, device=self.device, dtype=torch.float32)