#!/bin/bash
# Setup conda environment for butterfly classification on DGX
# Run: bash setup_env.sh

set -e

ENV_NAME="butterfly"

echo "=== Creating conda environment: $ENV_NAME ==="

# Check if env already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activating..."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate
source activate "$ENV_NAME" 2>/dev/null || conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA 12.4 compatible) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing dependencies ==="
pip install \
    timm>=1.0.0 \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    pillow \
    tqdm \
    tensorboard \
    grad-cam

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
import timm
print(f'timm: {timm.__version__}')
print('All good!')
"

echo "=== Done! Activate with: conda activate $ENV_NAME ==="
