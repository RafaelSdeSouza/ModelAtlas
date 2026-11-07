Five-minute quick start
=======================

Fit a small hexagonal map
-------------------------

Use a fixed random seed and supply features in comparable units. This toy
sample is generated in memory. The two components of each prototype describe
the original two inputs.

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from modelatlas import fit_som, plot_metric, plot_som_architecture, save_figure

   rng = np.random.default_rng(42)
   x = rng.normal(size=(260, 2)) @ [[1.0, 0.6], [0.0, 0.8]]
   x = (x - x.mean(axis=0)) / x.std(axis=0)
   som = fit_som(
       x, (5, 6), topology="hexagonal", steps=1800, seed=19,
       feature_labels=("Feature A", "Feature B"),
   )

   cell = tuple(som.best_matching_units(x[:1])[0])
   fig, ax = plot_som_architecture(som, highlight=cell, cmap="cividis")
   save_figure(fig, "outputs/hex_som")
   plt.close(fig)

   fig, ax = plot_metric(som, "u_matrix")
   print(som.diagnostics())
   plt.show()

Each prism is a fitted prototype. Color in the first figure gives its first
component. The U-matrix in the second figure averages fitted-space distances
to each unit's actual lattice neighbors. Read the two scales separately.

Use fitted weights
------------------

Wrap a codebook with shape ``(rows, columns, features)`` in
:class:`modelatlas.SOMMap`. Supply the training topology and transform any
observations into the same feature coordinates before assigning them. Use
:func:`modelatlas.som_from_minisom` for a fitted MiniSom object.

Compare several figures
-----------------------

For SOM plots with the same lattice, reuse ``projection`` and
``value_range``. For independently fitted prototype graphs, fit one
:func:`modelatlas.fit_prototype_layout` on the pooled codebooks in shared
feature coordinates. The :doc:`../guide/interpretation` describes what can be
compared in each setting.
