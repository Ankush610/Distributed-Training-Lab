
# Distributed Training Lab

Hands-on workshop covering distributed deep learning on HPC clusters using SLURM. Two datasets are used — **MNIST** (simple CNN, baseline familiarity with model classes/methods) and **CIFAR-10** (ResNet, full progression from single-GPU to DDP, FSDP, and Accelerate).

---

## Repository Structure

```
Distributed-Training-Lab/
│
├── setup/
│   ├── requirements.txt        # Python dependencies
│   ├── conda-pre-req.sh        # Conda activation helper (sourced by all SLURM scripts)
│   ├── dataset-cifar.py        # Download CIFAR-10 (run on login node)
│   └── dataset-mnist.py        # Download MNIST   (run on login node)
│
├── Torch/
│   ├── cifar-codes/
│   │   ├── 00_GpuOffload/      # Single-GPU baseline (ResNet + CIFAR-10)
│   │   ├── 01_Torch_DDP/       # PyTorch DistributedDataParallel
│   │   └── 02_Torch_FSDP/      # Fully Sharded Data Parallel
│   └── mnist-codes/
│       ├── 00_GpuOffload/      # Single-GPU baseline (CNN + MNIST)
│       └── 01_Torch_DDP/       # DDP on MNIST
│
├── Accelerate/
│   └── cifar-codes/
│       ├── 00_GpuOffload/      # Single-GPU via HuggingFace Accelerate
│       ├── 01_Accelerate_DDP/  # DDP via Accelerate
│       └── 02_Accelerate_FSDP/ # FSDP via Accelerate (with config YAML)
│
└── Kaggle-Colab/
    └── distributed-training-workshop.ipynb   # Cloud notebook version
```

Each numbered directory contains:
- A training script (`.py`)
- A SLURM job script (`script.sh`)
- A `logs/` directory where stdout/stderr are saved

---

## Setup

### 1. Install Miniconda (skip if already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
```

### 2. Load Conda into your shell

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Add this line to your `~/.bashrc` to make it permanent.

### 3. Create the environment

```bash
conda create -n tutorial python=3.11 -y
conda activate tutorial
```

### 4. Install dependencies

```bash
cd setup/
pip install -r requirements.txt
```

**Optional — JupyterLab** (if you want notebook support):

```bash
pip install jupyterlab
```

**Optional — uv** (faster pip alternative, install via conda-forge):

```bash
conda install -c conda-forge uv -y
```

---

## Download Datasets

> **Do this on the login node.** GPU compute nodes typically have no internet access.

```bash
# From the repo root
python setup/dataset-cifar.py   # downloads to Torch/cifar-codes/datasets/data-cifar
python setup/dataset-mnist.py   # downloads to Torch/mnist-codes/datasets/data-mnist
```

Both scripts accept `--data-dir /your/path` if you want to store datasets elsewhere.

---

## Running the Labs

Each lab directory has a `script.sh` you submit with `sbatch`. Before submitting, open the script and adjust:

- `--partition` — your cluster's GPU partition name
- `--gres=gpu:N` — number of GPUs to request
- `--nodes` / `--ntasks-per-node` — for multi-node jobs
- `--time` — wall-time limit

Then submit from inside the lab directory:

```bash
cd Torch/cifar-codes/01_Torch_DDP/
sbatch script.sh
```

Logs land in the `logs/` subdirectory:

```
logs/logs_<JOBID>.out   # stdout
logs/logs_<JOBID>.err   # stderr
```

Monitor your job:

```bash
squeue -u $USER
```

### Lab Progression

| Path | What it teaches |
|---|---|
| `Torch/mnist-codes/00_GpuOffload` | Building a CNN with PyTorch classes & training loop |
| `Torch/mnist-codes/01_Torch_DDP` | Adapting that model to multi-GPU with DDP |
| `Torch/cifar-codes/00_GpuOffload` | Single-GPU ResNet on CIFAR-10 (the baseline) |
| `Torch/cifar-codes/01_Torch_DDP` | DDP with `torchrun` |
| `Torch/cifar-codes/02_Torch_FSDP` | FSDP for memory-efficient sharded training |
| `Accelerate/cifar-codes/00_GpuOffload` | Same baseline via HuggingFace Accelerate |
| `Accelerate/cifar-codes/01_Accelerate_DDP` | DDP with minimal code changes using Accelerate |
| `Accelerate/cifar-codes/02_Accelerate_FSDP` | FSDP via Accelerate config YAML |

> **Note on batch size:** `--batch-size` in all scripts is **per GPU**, not total. With 2 GPUs and `--batch-size=512`, the effective global batch size is 1024. Keeping per-GPU batch size fixed while scaling GPUs is intentional.

---

## Conda Activation in SLURM Scripts

All `script.sh` files source `setup/conda-pre-req.sh` to activate the `tutorial` environment. This script auto-detects your Miniconda installation path. If conda is not in your `PATH` on the compute node, edit the fallback path in `setup/conda-pre-req.sh`:

```bash
CONDA_BASE_PATH="/home/apps/miniconda3"   # change to your cluster's miniconda path
```
