# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
from pathlib import Path

# sys.path.insert(0, os.path.abspath('../../GraphSlim'))
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

html_theme = "sphinx_rtd_theme"

source_suffix = [".rst", ".md"]
project = "GraphSlim"
version = release = "1.3.3"
copyright = "2024, Emory Melody Group"
author = "Shengbo Gong, Juntong Ni, Wei Jin"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = [
#     'myst_parser',
#     'sphinx.ext.mathjax',
#     'sphinx.ext.autodoc',
#     'sphinx.ext.napoleon',
#     'sphinx_rtd_theme',
#     'sphinx.ext.autosummary'
# ]

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
]

# Include constructor arguments and local helper methods in the API reference.
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "private-members": True,
    "special-members": "__getitem__, __repr__",
}

# Autodoc imports the real GraphSlim source, but does not run training. Mock
# optional/runtime libraries so Read the Docs needs only docs/requirements.txt.
autodoc_mock_imports = [
    "click",
    "gdown",
    "matplotlib",
    "networkx",
    "numpy",
    "skfuzzy",
    "sortedcontainers",
    "tqdm",
    "wandb",
    "torch",
    "torchvision",
    "texttable",
    "tensorboardX",
    "torch_geometric",
    "gensim",
    "node2vec",
    "deeprobust",
    "sklearn",
    "torch_sparse",
    "torch_scatter",
    "ogb",
    "pygsp",
    "scipy",
    "dgl",
    "networkit",
    "pyfpgrowth",
]

# autodoc_mock_imports = ['numpy', 'torch']
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"] if (Path(__file__).parent / "_static").is_dir() else []
html_css_files = ["edge-sparsification.css"]
epub_exclude_files = [".nojekyll"]
#
# add_module_names = False
#
# master_doc = 'index'
