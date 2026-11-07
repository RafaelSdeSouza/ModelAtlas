Supported models and adapters
=============================

The public ``plot_*`` functions return a Matplotlib figure and its primary
axes. Multi-panel functions return an array of axes. Training, alignment, and
inference have separate entry points. Choose an explicit renderer when you
need to set visual scales, titles or annotations for a paper.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Model or result
     - Inputs and supported scope
     - Views
   * - Rectangular or hexagonal SOM
     - :class:`modelatlas.SOMMap`, a fitted rectangular codebook array, or
       a fitted MiniSom via :func:`modelatlas.som_from_minisom`. The online
       :func:`modelatlas.fit_som` trainer uses finite, nonperiodic lattices.
     - Oblique prisms, input-to-prototype strands, top-down component planes,
       occupancy, quantization error, topographic error, U-matrix.
   * - Adaptive prototype graph
     - :class:`modelatlas.PrototypeGraph` with undirected, loop-free fitted
       adjacency. The supplied GNG-style fitter is experimental.
     - Projected nodes, learned edges, optional assignment occupancy and
       usable-spectroscopy fraction. :class:`modelatlas.PrototypeLayout`
       supplies shared coordinates and distortion diagnostics.
   * - Regularized GTM
     - :class:`modelatlas.GTMModel` with Gaussian radial-basis mapping,
       uniform latent weights, and isotropic precision.
     - :func:`modelatlas.plot_gtm` displays a codebook component on the
       fixed latent grid. ``responsibilities`` gives native soft memberships
       and ``assign`` gives hard MAP assignments.
   * - Dense fitted MLP
     - Fitted scikit-learn weight matrices with compatible layer sizes and
       small layers. Structural view handles multiple output columns.
     - :func:`modelatlas.plot_mlp` displays signed weights.
   * - Local MLP explanation
     - Binary logistic-output MLP with tanh, ReLU, logistic, or identity
       hidden activations. The oblique renderer uses two hidden layers.
     - :func:`modelatlas.explain_mlp` gives path attributions for one
       baseline contrast. :func:`modelatlas.plot_explanation_atlas` renders
       the chosen observation on a fixed or reference-fitted grid.
   * - Independent-network alignment
     - Binary logistic-output tanh MLPs with identical widths and ordered
       class labels. The oblique pair and residual views use two hidden layers.
     - Signed-permutation matching, paired oblique networks, held-out
       residual atlas and separate predictive-agreement diagnostics.
   * - Regression and GLM
     - A single coefficient vector or fitted estimator exposing one equation
       via ``coef_``, e.g. Ridge, PoissonRegressor or binary LogisticRegression.
       For a statsmodels fit, pass its parameter vector and explicit labels.
     - Signed coefficients on the specified link scale and observed versus
       predicted values. Supply interval endpoints and their interpretation.
   * - Photo-z partial identification
     - Fitted SOM, target features, target-aligned spectroscopic redshifts
       with NaN for unavailable values, and optional attempt indicators.
       For other hard partitions use :func:`modelatlas.partition_tail_bounds`.
     - Coverage and selection planes, cell and target identified intervals,
       radius sensitivity, and width contributions.

``visualize`` dispatches fitted objects from the first seven structural or
diagnostic families where a single default view is well defined. Call the
explanation and alignment functions directly to provide their comparison
data and baseline.

PyTorch, TensorFlow, convolutional and attention layers, causal models,
arbitrary probabilistic graphs, and multiclass path attribution have no
adapter in this preview. A rectangular SOM wrapped as hexagonal changes the
meaning of the U-matrix and topographic error. Record the fitted topology
with the weights and use the same feature preprocessing for all assignments.

The legacy :func:`modelatlas.plot_graph_atlas` also returns a diagnostics
dictionary as its third element. Use :func:`modelatlas.plot_prototype_graph`
for a figure and axes only, and :meth:`modelatlas.PrototypeLayout.diagnostics`
for a separate distortion report.
