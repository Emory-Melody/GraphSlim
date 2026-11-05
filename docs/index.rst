Welcome to GraphSlim's documentation!
=====================================

.. role:: edge-sparsification

A comprehensive package for node sparsification, :edge-sparsification:`edge sparsification`, coarsening and condensation based on PyG.

GraphSlim provides **49 graph reduction methods**: 5 node sparsifiers,
**18 edge sparsifiers**, 10 coarsening methods, and 16 condensation methods.
The sparsification family contains 23 methods in total. Counts refer to
canonical registry entries; aliases are not counted separately.

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation.md

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart.md

.. toctree::
   :hidden:

   source/modules
   changes_since_gc4nc
   testing_snapshot

.. toctree::
   :maxdepth: 1
   :caption: Dataset

   source/graphslim.dataset

.. toctree::
   :maxdepth: 1
   :caption: Model

   source/graphslim.models

.. toctree::
   :maxdepth: 1
   :caption: Method

   source/graphslim.sparsification
   source/graphslim.coarsening
   source/graphslim.condensation


.. toctree::
   :maxdepth: 1
   :caption: Evaluation

   source/graphslim.evaluation

.. toctree::
   :maxdepth: 1
   :caption: Package API

   source/graphslim
   source/graphslim.reduction
