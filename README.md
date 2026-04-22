# Distributed Training Lab

A hands-on workshop covering modern distributed deep learning techniques — from single-GPU baseline to fully sharded model parallelism — using **PyTorch** and **Hugging Face Accelerate** on a SLURM cluster.

The lab is split into two tracks:

- **MNIST track** (`mnist-codes/`) — lightweight intro to distributed concepts on a simple CNN. Great for understanding the mechanics of DDP without long runtimes.
- **CIFAR-10 / ResNet-50 track** (`cifar-codes/`) — production-grade benchmarking on a real architecture. Use this track to observe meaningful throughput differences, GPU memory savings from FSDP, and the effect of sharding strategies.

> **Why two tracks?** MNIST is too small and fast to expose real throughput bottlenecks — a simple CNN saturates in seconds and communication overhead dominates. ResNet-50 on CIFAR-10 is large enough that you can actually measure samples/sec gains, memory reduction per GPU, and gradient synchronization cost across strategies.

Each CIFAR module builds on the previous one so you can directly compare accuracy, throughput, and GPU memory across strategies.

---

## Curriculum

### MNIST Track — Conceptual Introduction

| # | Module | Strategy | Key API |
|---|--------|----------|---------|
| 00 | `mnist-codes/00_GpuOffload` | **Single GPU baseline** | `torch.cuda` |
| 01 | `mnist-codes/01_Torch_DDP` | **PyTorch DDP** | `DistributedDataParallel` |

### CIFAR-10 / ResNet-50 Track — Throughput Benchmarking

| # | Module | Strategy | Key API |
|---|--------|----------|---------|
| 00 | `cifar-codes/00_GpuOffload` | **Single GPU baseline** | `torch.cuda` |
| 01 | `cifar-codes/01_Torch_DDP` | **PyTorch DDP** | `DistributedDataParallel` |
| 02 | `cifar-codes/02_Accelerate_DDP` | **Accelerate DDP** | `Accelerator` |
| 03 | `cifar-codes/03_Torch_FSDP` | **PyTorch FSDP** | `FullyShardedDataParallel` |
| 04 | `cifar-codes/04_Accelerate_FSDP` | **Accelerate FSDP** | `Accelerator` + `custom_config.yaml` |

> **Notebook users:** Open `distributed-training-workshop.ipynb` for an interactive walkthrough of the full CIFAR-10 track, runnable end-to-end.

---

## Setup

### 1. Install dependencies
```bash
pip install -r setup/requirements.txt
```

### 2. Download datasets (run once on the login node)

**CIFAR-10** (used by the ResNet track):
```bash
python setup/dataset-cifar.py --data-dir /path/to/shared/datasets/data-cifar
```

**MNIST** (used by the introductory track):
```bash
python setup/dataset-mnist.py --data-dir /path/to/shared/datasets/data
```

### 3. Conda environment (SLURM clusters)

`setup/conda-pre-req.sh` auto-detects your conda installation and activates the `tutorial` environment. It is sourced by every SLURM script — you don't need to touch it unless your conda base path differs from the cluster default.

---

## Running on SLURM

Every module has a `script.sh` (or `script-ddp.sh`) ready for `sbatch`. Always `cd` into the module directory first so relative paths resolve correctly.

### MNIST Track

```bash
# Single GPU baseline
cd mnist-codes/00_GpuOffload && sbatch script.sh

# PyTorch DDP (2 GPUs)
cd mnist-codes/01_Torch_DDP && sbatch script-ddp.sh
```

### CIFAR-10 / ResNet-50 Track

```bash
# Single GPU baseline
cd cifar-codes/00_GpuOffload && sbatch script.sh

# PyTorch DDP (2 GPUs)
cd cifar-codes/01_Torch_DDP && sbatch script-ddp.sh

# Accelerate DDP (2 GPUs)
cd cifar-codes/02_Accelerate_DDP && sbatch script.sh

# PyTorch FSDP (2 GPUs)
cd cifar-codes/03_Torch_FSDP && sbatch script.sh

# Accelerate FSDP (2 GPUs, reads custom_config.yaml)
cd cifar-codes/04_Accelerate_FSDP && sbatch script.sh
```

Logs land in `./logs/logs_<jobid>.out` inside each module directory.

---

## Key Concept: Batch Size Is Per-GPU

> This is the single most common source of accuracy degradation when scaling to multiple GPUs.

PyTorch's `DataLoader` `batch_size` argument is **per GPU**, not total. `DistributedSampler` splits the *dataset* across GPUs automatically, but it never touches the batch size. If you request `batch_size=512` with 4 GPUs, your effective batch becomes **2048** — but your LR is tuned for 512, causing gradient steps to overshoot and loss to explode.

**The fix** (applied consistently across all modules):

```python
batch_size_per_gpu = args.batch_size // world_size   # or accelerator.num_processes

DataLoader(..., batch_size=batch_size_per_gpu, ...)
```

Pass `--batch-size=512` to mean "total effective batch of 512 across all GPUs". The code handles the division internally.

> **Accelerate note:** `accelerator.prepare()` wraps the sampler automatically, but it does *not* divide the batch size. The fix is identical — divide before constructing the `DataLoader`.

---

## Module Details

### MNIST Modules

#### 00 — Single GPU Baseline (MNIST)
Simple CNN trained on MNIST on a single GPU. No distributed setup. Use this to understand the training loop before any parallelism is introduced.

#### 01 — PyTorch DDP (MNIST)
The same CNN wrapped in `DistributedDataParallel` and launched with `torchrun`. Gradients are averaged across GPUs after each backward pass. Because MNIST+CNN is lightweight, you won't see a throughput win here — the value is in reading the minimal DDP boilerplate before encountering it at scale in the CIFAR track.

---

### CIFAR-10 / ResNet-50 Modules

#### 00 — Single GPU Baseline
Straightforward single-process ResNet-50 training on CIFAR-10. No distributed setup. Use this as your accuracy and timing reference point.

#### 01 — PyTorch DDP
Manual distributed setup via `torchrun` + `dist.init_process_group`. Gradients are averaged across GPUs after each backward pass. Each GPU holds a full model replica. With ResNet-50 you'll observe a real throughput increase here.

#### 02 — Accelerate DDP
Functionally equivalent to Module 01, but Accelerate handles the process group, device placement, and sampler setup. Your training code becomes nearly identical to single-GPU code.

#### 03 — PyTorch FSDP
Parameters, gradients, and optimizer states are sharded across GPUs. Each GPU holds only `1/world_size` of the model at rest — parameters are gathered on-the-fly during the forward/backward pass. Supports three sharding strategies via `--sharding-strategy`:

| Strategy | Shards | Use case |
|----------|--------|----------|
| `FULL_SHARD` | params + grads + optimizer state | Maximum memory savings |
| `SHARD_GRAD_OP` | grads + optimizer state only | Balance memory vs. comms overhead |
| `NO_SHARD` | nothing (equivalent to DDP) | Baseline comparison |

Optional flags: `--mixed-precision` (fp16), `--cpu-offload` (moves params to RAM between steps).

#### 04 — Accelerate FSDP
Same FSDP semantics as Module 03, but driven by `custom_config.yaml` instead of manual `FSDP(...)` constructor arguments. Accelerate reads the config at launch time via `--config_file`. Edit the YAML to change sharding strategy, mixed precision, or wrap policy without touching Python code.

---

## Repository Structure

```
Distributed-Training-Lab/
├── mnist-codes/
│   ├── 00_GpuOffload/             # Single GPU baseline (MNIST)
│   │   ├── mnist-train.py
│   │   └── script.sh
│   └── 01_Torch_DDP/              # PyTorch DDP (MNIST)
│       ├── mnist-train-ddp.py
│       └── script-ddp.sh
├── cifar-codes/
│   ├── 00_GpuOffload/             # Single GPU baseline (ResNet-50 / CIFAR-10)
│   │   ├── resnet-train.py
│   │   └── script.sh
│   ├── 01_Torch_DDP/              # PyTorch native DDP
│   │   ├── resnet-train-ddp.py
│   │   └── script-ddp.sh
│   ├── 02_Accelerate_DDP/         # HF Accelerate DDP
│   │   ├── resnet-train-accelerate-ddp.py
│   │   └── script.sh
│   ├── 03_Torch_FSDP/             # PyTorch native FSDP
│   │   ├── resnet-train-fsdp.py
│   │   └── script.sh
│   └── 04_Accelerate_FSDP/        # HF Accelerate FSDP
│       ├── resnet-train-accelerate-fsdp.py
│       ├── custom_config.yaml
│       └── script.sh
├── datasets/
│   ├── data/MNIST/                # MNIST raw binary files
│   └── data-cifar/                # CIFAR-10 batches
├── setup/
│   ├── conda-pre-req.sh           # Conda activation for SLURM jobs
│   ├── dataset-cifar.py           # CIFAR-10 downloader
│   ├── dataset-mnist.py           # MNIST downloader
│   └── requirements.txt           # Python dependencies
└── distributed-training-workshop.ipynb
```

---

## Learning Objectives

- Understand the difference between **data parallelism** (DDP) and **model sharding** (FSDP)
- Know why batch size must be divided per GPU and what happens when it isn't
- Use `torchrun` and `accelerate launch` to launch multi-GPU jobs under SLURM
- Configure FSDP sharding strategy, mixed precision, and CPU offload
- Swap between native PyTorch and HF Accelerate APIs for the same underlying strategy
- Understand why model architecture matters for benchmarking — and why MNIST alone doesn't tell the full story

Distributed Training PPT : [click here](https://drive.google.com/file/d/1uSDa7Zc3RMNRt1h45YHx1itKzr98uVx5/view?usp=sharing)

Happy training! ⚡
