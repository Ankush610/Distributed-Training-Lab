#!/bin/bash
# Pre-requisite script for Conda environment activation
# Centralized to avoid changing paths in multiple Slurm scripts across the lab.

# Environment name you want to activate
ENV_NAME="tutorial"

# 1. Try to get Conda base path dynamically if conda is already in PATH
if command -v conda &> /dev/null; then
    CONDA_BASE_PATH=$(conda info --base)
    echo "[*] Dynamically detected Conda at: $CONDA_BASE_PATH"
else
    # 2. Fallback path if conda is NOT in PATH (Cluster-specific default)
    CONDA_BASE_PATH="/home/apps/miniconda3"
    echo "[!] 'conda' command not found in PATH. Using fallback: $CONDA_BASE_PATH"
fi

# 3. Activate the environment
if [ -n "$CONDA_BASE_PATH" ] && [ -f "$CONDA_BASE_PATH/bin/activate" ]; then
    source "$CONDA_BASE_PATH/bin/activate" "$ENV_NAME"
    echo "[✓] Activated conda environment: $ENV_NAME"
else
    echo "[X] Error: Conda activate script not found at $CONDA_BASE_PATH/bin/activate"
    echo "    Please ensure conda is loaded (e.g., 'module load miniconda3') or update the fallback path in setup/conda-pre-req.sh."
    exit 1
fi
