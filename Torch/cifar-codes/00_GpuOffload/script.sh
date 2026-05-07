#!/bin/bash
#SBATCH --job-name=cifar_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=02:00:00
#SBATCH --reservation=cdac-app

echo "===================================="
echo "Job Name   : $SLURM_JOB_NAME"
echo "Node       : $SLURM_NODELIST"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/Node  : 1"
echo "Starting Single-GPU Training"
echo "===================================="

source $SLURM_SUBMIT_DIR/../../setup/conda-pre-req.sh

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

srun python resnet-train.py \
    --epochs=10 \
    --batch-size=512 \
    --lr=0.1 \
    --num-workers=8

echo "===================================="
echo "Training Completed"
echo "===================================="
