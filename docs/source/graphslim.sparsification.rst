graphslim.sparsification package
====================================

.. role:: edge-sparsification

The sparsification family contains five node selectors and
:edge-sparsification:`seven edge sparsifiers`.
Choose the canonical name with ``-M`` or ``create_reducer``:

.. list-table:: Node selectors (5)
   :header-rows: 1
   :widths: 20 50 30

   * - Type
     - Canonical method names
     - Meaning of ``-R``
   * - Node selection (5)
     - ``random``, ``kcenter``, ``herding``, ``cent_d``, ``cent_p``
     - Fraction of labeled training nodes used for the node budget.

.. list-table:: Edge sparsifiers (7)
   :header-rows: 1
   :widths: 20 50 30
   :class: edge-sparsification

   * - Type
     - Canonical method names
     - Meaning of ``-R``
   * - Edge sparsification (7)
     - ``random_edge``, ``spanning_forest``, ``g_spar``, ``local_degree``,
       ``rank_degree``, ``scan``, ``t_spanner``
     - Target fraction of input edges retained, except for the two methods below.

Node selectors choose existing nodes.

.. container:: edge-sparsification

   Edge sparsifiers retain the input node set, features, and labels while pruning
   connectivity. Their input is the full graph in the transductive setting and
   the training subgraph in the inductive setting. ``spanning_forest`` determines
   its output from a spanning forest, and ``t_spanner`` uses ``--ts`` to set
   stretch; both ignore ``-R``. The method determines ``args.edge_reduction``
   automatically. ``--edge_reduction`` and ``--node_reduction`` only assert the
   chosen method's type.

Node Sparsification (Coreset)
---------------------------------

graphslim.sparsification.coreset\_base module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.coreset_base
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.model\_based\_coreset\_base module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.model_based_coreset_base
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.herding module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.herding
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.kcenter module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.kcenter
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.kcenter\_sample module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.kcenter_sample
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.model\_free\_coreset\_base module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.model_free_coreset_base
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.random module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.random
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.cent\_degree module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.cent_degree
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.cent\_pagerank module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.cent_pagerank
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.herding\_agg module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.herding_agg
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.kcenter\_agg module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.kcenter_agg
   :members:
   :undoc-members:
   :show-inheritance:

.. rst-class:: edge-sparsification

Edge Sparsification
-----------------------

graphslim.sparsification.edge\_sparsification\_base module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.edge_sparsification_base
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.random\_edge module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.random_edge
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.g\_spar module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.g_spar
   :members:
   :undoc-members:
   :show-inheritance:


graphslim.sparsification.local\_degree module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.local_degree
   :members:
   :undoc-members:
   :show-inheritance:


graphslim.sparsification.rank\_degree module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.rank_degree
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.scan module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.scan
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.spanning\_forest module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.spanning_forest
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.t\_spanner module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.t_spanner
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.sparsification.utils module
-----------------------------------------

.. automodule:: graphslim.sparsification.utils
   :members:
   :undoc-members:
   :show-inheritance:
