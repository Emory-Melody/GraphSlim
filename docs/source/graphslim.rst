graphslim package
=====================

Subpackages
---------------

* :doc:`graphslim.dataset`
* :doc:`graphslim.models`
* :doc:`graphslim.sparsification`
* :doc:`graphslim.coarsening`
* :doc:`graphslim.condensation`
* :doc:`graphslim.evaluation`
* :doc:`graphslim.reduction`

Submodules
--------------

graphslim.config module
---------------------------

.. automodule:: graphslim.config
   :members:
   :undoc-members:
   :show-inheritance:

.. py:function:: cli(ctx, **kwargs)

   Parse CLI options, validate the reduction method, and create the run
   configuration. This callback is wrapped by Click; use
   :func:`graphslim.config.get_args` to obtain its return value from Python.

   :param ctx: Active Click context used to identify explicit CLI options.
   :param kwargs: Parsed command line option values.
   :returns: Configuration namespace for reduction and evaluation.

graphslim.compat module
---------------------------

.. automodule:: graphslim.compat
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.doctor module
---------------------------

.. automodule:: graphslim.doctor
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.hpo module
------------------------

.. automodule:: graphslim.hpo
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.run_eval module
-----------------------------

.. automodule:: graphslim.run_eval
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.tracking module
-----------------------------

.. automodule:: graphslim.tracking
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.visualization module
----------------------------------

.. automodule:: graphslim.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Internal import helpers
---------------------------

.. automodule:: graphslim._lazy
   :members:

graphslim.train\_all module
-------------------------------

.. automodule:: graphslim.train_all
   :members:
   :undoc-members:
   :show-inheritance:

graphslim.utils module
--------------------------

.. automodule:: graphslim.utils
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
-------------------

.. automodule:: graphslim
   :members:
   :undoc-members:
   :show-inheritance:
