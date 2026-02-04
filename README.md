# FedCompass+: A Hybrid Classical-Quantum Federated Learning Framework

The code provides a comprehensive simulation of Federated Learning (FL) on non-IID data (Label-shift) using both classical and hybrid quantum-classical models. It includes the proposed method (**FedCompass+**, referenced as `our_FL.py`) as well as baselines like **Scaffold** and **FedNova**.

### 🗂️ Directory & File Details

* **`AblationStudy/`**: Contains scripts to verify the contribution of specific modules within the FedCompass framework.
* **`Label-shift/`**: The main root for experiments focusing on the "Label Shift" non-IID data challenge.
* **`Feature-shift/`**: The main root for experiments focusing on the "Feature-shift" non-IID data challenge.
* **`CIFAR-10/`**: Dataset.
* **`alpha_0.7/`**: Represents the degree of data heterogeneity. Data is partitioned among clients using a Dirichlet distribution with . (Lower  indicates higher heterogeneity).
* **`lenet_pqc`**: A hybrid model combining a classical LeNet backbone with a Variational Quantum Circuit (VQC/PQC) layer.
* **`resnet18_qcnn`**: A deeper hybrid model integrating Quantum Convolutional Neural Network (QCNN) layers into ResNet18.


* **Main Entry(`algorithms/`)**:
* **`FedAvg.py`**: Script to start the FL simulation (FedAvg).
* **`Ours.py`**: Script to start the FL simulation (FedCompass+).


* **Data Process(`data_process/`)**:
* **`data_utils.py`**: Data loading & Dirichlet partitioning utils.


* **Model Architectures(`models/`)**:
* **`modelx.py`**: Model definition (Hybrid Quantum-Classical Network).


* **Core Algorithms (`federated/`)**:
* `our_FL.py`: **FedCompass**. The core implementation of our proposed hybrid classical-quantum federated learning framework.
* `Scaffold_FL.py`: Implementation of the Scaffold algorithm (Control Variates).
* `FedNova_FL.py`: Implementation of the FedNova algorithm (Normalized Averaging).


## 🛠️ Environment Setup

To reproduce the results, please create a virtual environment and install the specific dependencies as follows.

### 1. Create and Activate Conda Environment
We recommend using **Python 3.10**.

```bash
conda create -n env python=3.10
conda activate env

```

### 2. Install PyTorch (GPU Support)

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

### 3. Install Other Dependencies

Install the required libraries for Federated Learning (Flower), Quantum Computing (PennyLane), and other utilities:

```bash
pip install flwr==1.5.0 tqdm numpy pennylane "ray>=2.3.0" matplotlib pillow scikit-learn seaborn pandas opacus pyyaml tenseal kaggle tensorflow

```

> **Note:** Ensure that your system supports CUDA 12.1 for the PyTorch installation. If you are using a different CUDA version, please adjust the `--index-url` in step 2 accordingly.

```

## 🚀 How to Run

The experiments are segmented by the directory structure. To run a specific experiment (e.g., the proposed method on MNIST with alpha=0.7 using LeNet-PQC), navigate to the corresponding folder or run the script from the root.

### Running the Proposed Method (FedCompass+)

```bash
# Example: Run FedCompass+ with LeNet-PQC on MNIST (alpha=0.7)
python Label-shift/MNIST/alpha_0.7/lenet_pqc/federated/our_FL.py

```

### Running Baselines

You can compare the performance with baseline algorithms provided in the same directory:

```bash
# Run Scaffold Baseline
python Label-shift/MNIST/alpha_0.7/lenet_pqc/federated/Scaffold_FL.py

# Run FedNova Baseline
python Label-shift/MNIST/alpha_0.7/lenet_pqc/federated/FedNova_FL.py

```

### Ablation Studies

To reproduce the ablation study results:

```bash
python AblationStudy/AblationStudy1/algorithms/Ablation_1.py

```