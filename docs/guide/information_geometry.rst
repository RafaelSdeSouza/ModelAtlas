Information geometry and stable model neighborhoods
===================================================

This research-preview component is **independent of fitted-model architecture**.
It compares predictions from models that may use different parameters and
computation graphs. It does not equate visual proximity with comparable causal
mechanisms or with calibrated scientific conclusions.

Predictive geometry
-------------------

For binary classifiers evaluated on the same reference inputs :math:`x_1,\ldots,x_N`,
let :math:`p_m(x_i)=P_m(Y=1\mid x_i)`.
For Bernoulli distributions, the Fisher–Rao geodesic distance is

.. math::

   d_{\mathrm{FR}}(p,q)=2\left|\arcsin\sqrt p-\arcsin\sqrt q\right|.

We define a reference-weighted distance between fitted models

.. math::

   D^2(m,m')=\sum_{i=1}^{N}w_i\,d_{\mathrm{FR}}^2\big(p_m(x_i),p_{m'}(x_i)\big),
   \quad w_i\geq 0,\quad \sum_i w_i=1.

This defines a metric on distinct prediction vectors and a pseudometric
on fitted-model objects. Models whose predictions agree on the finite
reference sample have zero distance; that is not proof of global equivalence.
The 3D embedding is computed from the transformed prediction vectors using
classical metric multidimensional scaling. The viewing camera adds another
projection, so **pixel distances are not Fisher–Rao distances**.

Stable model relationships
--------------------------

The edges in the 3D atlas connect nearest neighbors in predictive geometry,
**not** actual neural-network weights or causal dependencies.
We repeatedly resample reference observations and recompute predictive
neighborhoods. Each edge's visual strength represents the fraction of
bootstrap resamples in which the edge appears.

This probes sensitivity to the *reference sample*, not training randomness,
measurement errors, calibration uncertainty, or model re-training.
For broader validity, a study should vary all relevant data-generation
mechanisms and explicitly report the chosen reference population.

Relationship to group actions and information lattices
------------------------------------------------------

An exact hidden-neuron permutation, accompanied by the corresponding
weight permutation, leaves neural-network predictions unchanged.
The predictive geometry is therefore invariant to this transformation.
By contrast, architecture-specific neuron diagrams may change.

A complementary information lattice studies what a fixed fitted model's
predictions still depend on when we observe progressively more features.
For nested sigma-algebras :math:`\mathcal F_S\subseteq\mathcal F_T`,

.. math::

   E\operatorname{Var}\big(f(X)\mid\mathcal F_T\big)
   \leq E\operatorname{Var}\big(f(X)\mid\mathcal F_S\big).

This is a direct consequence of conditional-expectation projections in
:math:`L^2`, not a novel theorem. It describes variation hidden by a
representation, not estimation uncertainty, identification, or causality.

Possible extensions (not implemented)
-------------------------------------

- **Optimal transport:** compare full predictive distributions with compatible
  response metrics and sampling assumptions rather than scalar probabilities.
- **Persistent neighborhoods:** test graph components across a range of
  distance thresholds, rather than choosing one nearest-neighbor scale.
- **Sheaf / local-to-global consistency:** test whether explanations or
  predictions fitted on overlapping data patches agree where patches overlap.
  A formal sheaf construction requires explicitly defined restriction maps;
  a collection of adjacent polygons is not a sheaf.

These approaches have mathematical precedents and require independent
methodological validation before claiming novelty.
