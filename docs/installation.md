# Installation

Use Python 3.10 or 3.11. Install PyTorch and matching `torch-scatter` /
`torch-sparse` wheels for your CPU or CUDA environment before installing GraphSlim.

```bash
git clone https://github.com/Emory-Melody/GraphSlim.git
cd GraphSlim
pip install -e .
```

See the repository [README](https://github.com/Emory-Melody/GraphSlim#install)
for the full CPU setup and optional dependency groups. For a published version,
use `pip install graphslim`. The `uv.lock` environment is consumed by
`uv sync --locked`; PyG extension wheels must match its Torch version.

See [runtime compatibility](compatibility.md) for tested CPU/CUDA combinations,
optional DGL graph interchange and the real-operator self-check:

```bash
python -m graphslim.doctor --device cpu --dgl
```

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
