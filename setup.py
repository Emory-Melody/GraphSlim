# import os
# import sys
#
# if os.path.abspath('..') not in sys.path:
#     sys.path.append(os.path.abspath('..'))
from setuptools import find_packages, setup

name = 'graphslim'
requires_list = [
    'click',
    'deeprobust',
    'gdown',
    'networkit',
    'networkx',
    'numpy',
    'ogb',
    'PyGSP',
    'scikit_learn',
    'scipy',
    'sortedcontainers',
    'torch',
    'torch_geometric',
    'tqdm',
]

setup(
    name=name,
    version='1.1.1',
    author="Rockcor",
    author_email='jshmhsb@gmail.com',
    description="Slimming the graph data for graph learning",
    packages=find_packages(),
    python_requires='>=3.7',
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    license="MIT",
    install_requires=requires_list
)
