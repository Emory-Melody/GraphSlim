# Installation

Use Python 3.10 or 3.11. Install PyTorch and matching `torch-scatter` /
`torch-sparse` wheels for your CPU or CUDA environment before installing GraphSlim.

The following CPU setup uses the same DGL/NumPy constraints as CI:

```bash
git clone https://github.com/Emory-Melody/GraphSlim.git
cd GraphSlim
git checkout --detach 1.3.5
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-scatter torch-sparse --no-deps --no-index -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
uv pip install -e ".[dev,dgl,fuzzy]" -c constraints/cpu-torch26.txt
python -m graphslim.doctor --device cpu --dgl
```

`constraints/cpu-common.txt` pins DGL 1.1.3 and NumPy 1.26.4; the Torch-specific
constraint files match Torch 2.6/torchvision 0.21 or Torch 2.8/torchvision 0.23.
For the latter, use the matching Torch/PyG wheels and `cpu-torch28.txt`.
The optional `.[viz]` extra enables local graph visualization.

For CUDA, select matching Torch and PyG extension wheels using the
[README GPU setup](https://github.com/Emory-Melody/GraphSlim/blob/1.3.5/README.md#gpu-setup-and-checks)
and keep the chosen Torch version constrained. GPU execution uses native PyTorch operations. Check it with
`python -m graphslim.doctor --device cuda:0 --dgl`.

The source version above includes 49 canonical methods and eight GNN backbones.
PyPI still distributes the earlier 1.2.1 (checked 2026-11-06), so use the Git tag
for these features. `uv sync --locked` consumes the separate dependency lock;
PyG extension wheels must match the Torch runtime it resolves. The versioned CPU
recipe above and its constraints describe the tested CPU environment.

## Build the documentation

The website can be built in a separate Python 3.10 or 3.11 environment:

```bash
python -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
sphinx-build -E -W --keep-going -b html -d docs/_build/doctrees docs docs/_build/html
```

Open `docs/_build/html/index.html` to preview the site. Sphinx reads the package
source and mocks runtime dependencies, so this build needs no PyTorch, CUDA,
DGL or MKL installation. It documents the API without exercising those libraries.
The independent documentation CI job also builds EPUB and LaTeX sources;
Read the Docs compiles the latter into its downloadable PDF.
