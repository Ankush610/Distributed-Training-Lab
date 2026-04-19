#!/bin/bash
#SBATCH --job-name=cifar_accelerate_ddp
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

echo "===================================="
echo "Job Name   : $SLURM_JOB_NAME"
echo "Master Addr: $MASTER_ADDR"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/Node  : $GPUS_PER_NODE"
echo "Starting CIFAR-10 ResNet-50 Accelerate DDP Training"
echo "===================================="

source $SLURM_SUBMIT_DIR/../../setup/conda-pre-req.sh

srun accelerate launch \
  --num_processes $GPUS_PER_NODE \
  resnet-train-accelerate-ddp.py \
  --epochs=10 \
  --batch-size=512 \
  --lr=0.1 \
  --num-workers=8

echo "===================================="
echo "Training Completed"
echo "===================================="
