#!/bin/bash

# User-defined variables
PT_VERSION=1.13.1   # PyTorch version
CU_VERSION=cu116    # CUDA version
PYTHON_VERSION=cp38 # Python version

# Create a directory for wheel files
mkdir -p ~/pytorch_wheels
cd ~/pytorch_wheels

# Download the wheel files
wget https://data.pyg.org/whl/torch-${PT_VERSION}+${CU_VERSION}/torch_scatter-2.1.0+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${PT_VERSION}+${CU_VERSION}/torch_sparse-0.6.15+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${PT_VERSION}+${CU_VERSION}/torch_cluster-1.6.0+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-${PT_VERSION}+${CU_VERSION}/torch_spline_conv-1.2.1+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl

# Install the downloaded wheels
pip install torch_scatter-2.1.0+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
pip install torch_sparse-0.6.15+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
pip install torch_cluster-1.6.0+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
pip install torch_spline_conv-1.2.1+pt${PT_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-linux_x86_64.wh
