Research baseline and compatibility
===================================

The development branch is stacked on draft PR #1, commit
``5041936300e5660685b091e940796bd515326dd9``. Its 36 original tests passed
before any changes. The research branch is the baseline for numerical
experiments and the historical interpretation notes.

.. list-table::
   :header-rows: 1
   :widths: 23 40 37

   * - Baseline area
     - Existing entry points and behavior
     - Scope and changes in this branch
   * - SOM
     - ``SOMMap``, ``train_som``, ``som_from_minisom``, ``plot_som``,
       ``plot_metric``, ``plot_som_architecture``. Rectangular trainer,
       four-connected diagnostics, vector drawings with fixed affine layout.
     - Keep the rectangular trainer and codebook array return. Add a typed
       topology, true hexagonal training and diagnostics, explicit camera
       geometry and shared color limits. Oblique drawing coordinates change.
   * - Adaptive/GTM
     - ``PrototypeGraph``, ``GTMModel``, ``growing_neural_gas``,
       ``train_gtm``, ``partition_tail_bounds``, ``plot_graph_atlas``.
       GNG-style reconnection of isolated vertices and a regularized GTM fit.
     - Retain fitted graph construction and synthetic benchmarks. Expose
       soft GTM responsibilities and an explicit reusable graph layout.
       The graph figure now uses unwhitened PCA and an orthographic camera.
   * - Photo-z
     - ``PhotoZCells``, ``PhotoZBounds``, ``sharp_tail_bounds``,
       ``plot_photoz_map``, ``plot_cell_plane``,
       ``plot_photoz_diagnostics``, ``plot_photoz_architecture``.
     - Retain the conditional empirical-reference model. Carry topology
       through diagnostics, clarify strict-tail endpoint behavior, validate
       partition input and plot the population sensitivity curve.
   * - Neural structures and XAI
     - ``plot_mlp``, ``PathExplanation``, ``explain_mlp``,
       ``activation_signatures``, ``plot_explanation_atlas``. Small
       scikit-learn MLPs and local scalar-logit attributions.
     - Expose reference layouts and shared attribution scales. Keep the
       numerical edge-conductance decomposition and baseline semantics.
   * - Symmetry
     - ``Alignment``, ``match_tanh_mlp``,
       ``transform_signed_permutation``, ``representational_metrics``,
       ``plot_independent_pair``, ``plot_residual_atlas``,
       ``plot_alignment_diagnostic``.
     - Retain the signed Hungarian matching criterion and the original
       examples. Return live figures and axes from plotters, with optional
       export. Compare logits and probabilities to verify invariance.
   * - Regression and dispatch
     - ``plot_coefficients``, ``plot_predictions``, ``visualize``.
       One coefficient vector, evaluation sequences, model-specific dispatch.
     - Add binary GLM coefficient shapes and caller-supplied interval
       endpoints. Extend dispatch to fitted graphs and GTM.

The independent real-data MLP checks use a single train/test split and two
seeds per dataset. Adaptive photo-z conclusions are limited to the included
synthetic designs. Feature selection, preprocessing and simulation assumptions
remain part of each experiment's provenance.
