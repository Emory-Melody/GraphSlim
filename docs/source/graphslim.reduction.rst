graphslim.reduction package
===============================

The registry resolves canonical method names without importing every reducer.
Use ``create_reducer(method, setting, data, args)`` to construct a reducer and
``reducer.reduce(data)`` to populate ``feat_syn``, ``adj_syn``, and ``labels_syn``.
Node methods interpret ``reduction_rate`` as a fraction of labeled training
nodes. Edge methods retain the input nodes and apply the rate to edges, except
for spanning forests and stretch-controlled spanners.

graphslim.reduction.registry module
---------------------------------------

.. automodule:: graphslim.reduction.registry
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.reduction.lightweight module
------------------------------------------

.. automodule:: graphslim.reduction.lightweight
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.reduction.cluster_ops module
------------------------------------------

.. automodule:: graphslim.reduction.cluster_ops
   :members:
   :undoc-members:
