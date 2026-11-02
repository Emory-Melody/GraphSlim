graphslim.sparsification package
====================================

.. role:: edge-sparsification

The sparsification family contains **23 methods**: five node selectors and
:edge-sparsification:`18 edge sparsifiers`. These are part of GraphSlim's
49 canonical reduction methods; aliases are not counted separately.
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

.. list-table:: Edge sparsifiers (18)
   :header-rows: 1
   :widths: 20 50 30
   :class: edge-sparsification

   * - Type
     - Canonical method names
     - Meaning of ``-R``
   * - Edge sparsification (18)
     - ``random_edge``, ``spanning_forest``, ``g_spar``, ``local_degree``,
       ``rank_degree``, ``scan``, ``t_spanner``, ``simmelian_parametric``,
       ``simmelian_non_parametric``, ``quadrilateral_simmelian``,
       ``simmelian_multiscale``, ``degree_multiscale``, ``local_similarity``,
       ``multiscale``, ``random_node_edge``, ``forest_fire``, ``triangle``,
       ``algebraic_distance``
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

NetworKit backend coverage
--------------------------

All 15 concrete Python sparsifiers in ``networkit.sparsification`` are
available, plus the three existing Spanning Forest, Rank Degree, and
``t_spanner`` reducers. ``g_spar`` already uses Jaccard similarity; its alias
is not counted as another method. The generic ``LocalSparsifier`` can be
composed with any of the 15 backends using ``{"local": true}`` and is not
counted as a separate algorithm.

.. list-table:: Native backends and constructor settings
   :header-rows: 1
   :widths: 25 35 40

   * - GraphSlim method
     - NetworKit class
     - Constructor defaults / options
   * - ``random_edge``
     - ``RandomEdgeSparsifier``
     - Upstream defaults.
   * - ``g_spar``
     - ``JaccardSimilaritySparsifier``
     - Upstream defaults; Python alias ``jaccard_similarity``.
   * - ``local_degree``
     - ``LocalDegreeSparsifier``
     - Upstream defaults.
   * - ``scan``
     - ``SCANSparsifier``
     - Upstream defaults.
   * - ``simmelian_parametric``
     - ``SimmelianSparsifierParametric``
     - ``maxRank=10``.
   * - ``simmelian_non_parametric``
     - ``SimmelianSparsifierNonParametric``
     - Upstream defaults.
   * - ``quadrilateral_simmelian``
     - ``QuadrilateralSimmelianSparsifier``
     - Upstream defaults.
   * - ``simmelian_multiscale``
     - ``SimmelianMultiscaleSparsifier``
     - Upstream defaults.
   * - ``degree_multiscale``
     - ``DegreeMultiscaleSparsifier``
     - ``degsToAttrValue="max"``; also min, sum, product, or a Python callable.
   * - ``local_similarity``
     - ``LocalSimilaritySparsifier``
     - Upstream defaults.
   * - ``multiscale``
     - ``MultiscaleSparsifier``
     - Upstream defaults.
   * - ``random_node_edge``
     - ``RandomNodeEdgeSparsifier``
     - ``above=True``; all input node IDs are retained.
   * - ``forest_fire``
     - ``ForestFireSparsifier``
     - ``burnProbability=0.5``, ``targetBurntRatio=1.0``.
   * - ``triangle``
     - ``TriangleSparsifier``
     - Triangle scores with the GraphSlim budget adapter described below.
   * - ``algebraic_distance``
     - ``AlgebraicDistanceSparsifier``
     - ``numberSystems=10``, ``numberIterations=30``, ``omega=0.5``, ``norm=0``.

Pass constructor options as a JSON object using ``--sparsifier_options`` or
as a ``"sparsifier_options"`` object in ``--config``. Explicit CLI values
replace the config object. Python callers may set the same attribute on
``args``, using a mapping (including a callable for ``degsToAttrValue``)::

   python -m graphslim.train_all -D cora -M forest_fire -R 0.1 -G -1 --run_reduction 1 --sparsifier_options '{"burnProbability": 0.4, "targetBurntRatio": 0.8}'

Native size parameterization targets a fraction of input edges. Threshold
methods can miss the requested count when scores tie; the selected adjacency
and ``reduction_metadata["reduced_edges"]`` give the actual count. Endpoints
``-R 0`` and ``-R 1`` return an empty and full edge set respectively, preserving
all nodes. No output edges are added outside the input graph.

Upstream ``TriangleSparsifier`` does not implement size parameterization.
GraphSlim ranks its native triangle counts and retains exactly
``floor(R * input_edges)`` edges. Equal counts are ordered by the canonical
endpoint pair. This budget adapter is recorded in
``reduction_metadata["budget_selection"]``. When ``local=true``, the native
local-filter parameterization applies instead.

Randomized backends seed NetworKit from ``args.seed``; repeatability depends
on a fixed thread count and backend version.

The existing reduction pipeline uses unweighted, undirected connectivity;
``edge_cutter`` also accepts weighted NetworKit graphs and returns retained
weights. The new methods have functional integration checks; they do not
extend the fixed 123-pair accuracy reproduction snapshot.

The upstream API is documented at
https://networkit.github.io/dev-docs/python_api/sparsification.html.

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

graphslim.sparsification.networkit_sparsifiers module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: graphslim.sparsification.networkit_sparsifiers
   :members:
   :undoc-members:
   :show-inheritance:
