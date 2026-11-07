API reference
=============

The functions and models below can be imported from ``modelatlas``. Plot
functions return Matplotlib objects for composition and export.

SOMs and camera geometry
------------------------

.. autoclass:: modelatlas.SOMLattice
   :members:

.. autoclass:: modelatlas.SOMMap
   :members:

.. autofunction:: modelatlas.fit_som
.. autofunction:: modelatlas.train_som
.. autofunction:: modelatlas.som_from_minisom
.. autofunction:: modelatlas.plot_som
.. autofunction:: modelatlas.plot_som_architecture
.. autofunction:: modelatlas.plot_metric

.. autoclass:: modelatlas.ObliqueProjection
   :members:

.. autoclass:: modelatlas.ProjectionDiagnostics

.. autofunction:: modelatlas.projection_diagnostics

Prototype graphs and GTM
------------------------

.. autoclass:: modelatlas.PrototypeGraph
   :members:

.. autoclass:: modelatlas.GTMModel
   :members:

.. autoclass:: modelatlas.PrototypeLayout
   :members:

.. autofunction:: modelatlas.growing_neural_gas
.. autofunction:: modelatlas.train_gtm
.. autofunction:: modelatlas.fit_prototype_layout
.. autofunction:: modelatlas.plot_prototype_graph
.. autofunction:: modelatlas.plot_graph_atlas
.. autofunction:: modelatlas.plot_gtm

Neural networks and explanations
--------------------------------

.. autofunction:: modelatlas.plot_mlp

.. autoclass:: modelatlas.PathExplanation
   :members:

.. autofunction:: modelatlas.explain_mlp
.. autofunction:: modelatlas.activation_signatures
.. autofunction:: modelatlas.explanation_layout
.. autofunction:: modelatlas.plot_explanation_atlas

.. autoclass:: modelatlas.Alignment

.. autofunction:: modelatlas.transform_signed_permutation
.. autofunction:: modelatlas.match_tanh_mlp
.. autofunction:: modelatlas.representational_metrics
.. autofunction:: modelatlas.plot_independent_pair
.. autofunction:: modelatlas.plot_residual_atlas
.. autofunction:: modelatlas.plot_alignment_diagnostic

Regression and photo-z
----------------------

.. autofunction:: modelatlas.plot_coefficients
.. autofunction:: modelatlas.plot_predictions

.. autoclass:: modelatlas.PhotoZCells
   :members:

.. autoclass:: modelatlas.PhotoZBounds
   :members:

.. autofunction:: modelatlas.sharp_tail_bounds
.. autofunction:: modelatlas.partition_tail_bounds
.. autofunction:: modelatlas.plot_cell_plane
.. autofunction:: modelatlas.plot_photoz_map
.. autofunction:: modelatlas.plot_photoz_diagnostics
.. autofunction:: modelatlas.plot_photoz_architecture
.. autofunction:: modelatlas.plot_identification_sensitivity

Dispatch and export
-------------------

.. autofunction:: modelatlas.visualize
.. autofunction:: modelatlas.save_figure
