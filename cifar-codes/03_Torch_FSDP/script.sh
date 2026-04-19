#!/bin/bash
#SBATCH --job-name=cifar_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=02:00:00
#SBATCH --reservation=cdac-app

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

echo "===================================="
echo "Job Name   : $SLURM_JOB_NAME"
echo "Master Addr: $MASTER_ADDR"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/Node  : $GPUS_PER_NODE"
echo "World Size : $WORLD_SIZE"
echo "Starting CIFAR-10 ResNet-50 FSDP Training"
echo "===================================="

source $SLURM_SUBMIT_DIR/../../setup/conda-pre-req.sh

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=$GPUS_PER_NODE \
    --rdzv-id=$SLURM_JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    resnet-train-fsdp.py \
        --epochs=10 \
        --batch-size=512 \
        --lr=0.1 \
        --num-workers=8 \
        --sharding-strategy=FULL_SHARD

echo "===================================="
echo "Training Completed"
echo "===================================="
