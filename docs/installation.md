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
