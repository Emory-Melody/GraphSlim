# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

# sys.path.insert(0, os.path.abspath('../../GraphSlim'))
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.append('../..')
sys.path.append('../../GraphSlim')

html_theme = "sphinx_rtd_theme"

source_suffix = ['.rst', '.md']
project = 'GraphSlim'
copyright = '2024, Melody Group'
author = 'Shengbo Gong, Juntong Ni, Wei Jin'

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

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'myst_parser',
              'sphinx.ext.autosummary', 'sphinx.ext.mathjax', 'sphinx.ext.viewcode', 'sphinx.ext.githubpages']

autodoc_mock_imports = ['torch', 'torchvision', 'texttable', 'tensorboardX',
                        'torch_geometric', 'gensim', 'node2vec', 'deeprobust',
                        'sklearn', 'torch_sparse', 'torch_scatter', 'ogb']

# autodoc_mock_imports = ['numpy', 'torch']
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
#
# add_module_names = False
#
# master_doc = 'index'
