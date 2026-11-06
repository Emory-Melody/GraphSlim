# Changelog

## 1.3.3 - 2026-11-06

- Use anomaly-class AP for Yelp/Amazon validation, HPO and final evaluation, including SGFormer and ClustGDD selection. Preserve historical macro-F1 records with their original labels.
- Reject HPO resumes when the grid, scientific configuration, seeds, model, graph, source or dependency versions differ; reject legacy histories without an identity.
- Remove the optional CUDA Graph backend and Streamlit/pyvis interface while retaining ordinary CUDA and local visualization.
- Restore verified Yelp AP results with source and uncertainty labels. Separate edge graph labels from node budgets.
- Correct the 22-method default performance roster and distinguish archived result tables from the cumulative reproduction artifact.
- Share tested CPU dependency constraints across installation instructions and CI, and add runnable reducer-extension and HPO guides.

## 1.3.2 - 2026-11-06

- Raise test coverage to 98.8% of statements and 96.1% of branches (1,589 tests),
  covering every reduction family, model, dataset loader and command-line entry point
  with small synthetic CPU graphs.
- Fix defects found by the new tests. Fixes that change results of registered methods:
  CentP's PageRank now transposes its transition matrix; inductive K-Center and Herding
  score training-graph embeddings; GDEM starts from Xavier features, keeps held-out
  nodes and labels out of its training map and caches eigenpairs under `--load_path`;
  GCSNTK's sparse Kronecker product no longer modifies coalesced inputs; SFGC's
  k-center initializer uses the released fixed selection where one is packaged.
- Fix paths that crashed or returned wrong values: HeavyEdge in the pipeline,
  inductive spectral coarsening, VNG with GCN teachers, graph sparsification and
  coarsening diagnostics, GEOM/SFGC expert snapshots on CPU, GEOM's learnable
  temperature and its Yelp/Amazon configs, SGDD on small graphs, GCDM with one layer,
  SGFormer options, GAT in training mode, dense batch normalization, evaluation and
  dataset utilities, tracking and visualization. Remove the unregistered, unrunnable
  Mirage module.
- Remove CentP and GDEM from the numerical reference comparison, because the fixed
  defects also affected the code behind their GC4NC references, and rerun the affected
  inductive K-Center and Herding settings. The new artifact
  `benchmark/results/reproducibility_20261106` embeds the unchanged 1.3.1 artifact:
  110 pairs, one-sided shortfall below 5% for 107/110 and below 10% for 110/110.
- Pin installation instructions to the `1.3.2` Git tag and record a release-specific
  test and coverage run in `validation/release_1_3_2`.

## 1.3.1 - 2026-11-05

- Freeze a distinct Git source release for the 49-method, eight-backbone package.
- Add license identifiers to all 106 package Python modules and 24 shell scripts,
  retaining bundled MIT, Apache-2.0 and BSD-3-Clause terms and upstream notices.
- Remove the obsolete manuscript draft from the source tree.
- Publish author-confirmed GC4NC reference corrections and a versioned offline
  result artifact. Numerical settings, comparison scores and results are unchanged.
- Record a fresh release-specific CPU test and coverage run, and pin installation
  instructions to the `1.3.1` Git tag.

Package Python module ASTs are unchanged from the preceding source version.
The historical `1.3.0` tag is retained at its original commit.
