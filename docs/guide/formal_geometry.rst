Formal model-agnostic geometry (research preview)
=================================================

Overview
--------

The functional atlas starts from fitted predictors :math:`f_m` evaluated on
**the same reference observations** :math:`x_i`. It intentionally does not
assume a common network architecture, kernel, graph, or parameter dimension.
Binary class probabilities should refer to the same class for every model;
regressors must be compared in a common output unit.

Groups and functional equivalence
---------------------------------

For a group :math:`G` acting on parameterizations :math:`\theta`, a
*function-preserving* action obeys

.. math::

    f_{g\theta}(x)=f_\theta(x),\qquad g\in G.

Hence any model comparison that depends only on the predictive function
is automatically invariant on group orbits. The quotient of parameters
under a **known** group action is not always the same as the set of all
functionally equivalent parameterizations.

For model families that cannot be compared in parameter space, define an
empirical functional pseudometric on the common reference distribution:

.. math::

    d^2_{\hat P}(m,m')=\frac{1}{N}\sum_{i=1}^{N}
       \left[f_m(x_i)-f_{m'}(x_i)\right]^2.

The code uses weighted classical multidimensional scaling on prediction
profiles. Distance is a *pseudometric on fitted model objects*: distinct
models may predict identically on the reference observations. Equality
there is **not proof** of global functional equivalence. Both 2D and 3D fractions of between-model squared distance retained and
corresponding stress diagnostics are reported. The oblique camera displaying
3D coordinates is itself not a distance-preserving projection.

Fibers of a representation
--------------------------

For a measurable compression :math:`h:\mathcal X\to\mathcal H`, the fiber
:math:`h^{-1}(u)` contains all inputs indistinguishable under :math:`h`.
The **conditional distribution** of :math:`f(X)` given :math:`h(X)=u` is a
statistical object on this fiber, where a suitable regular conditional
probability exists. The example approximates these conditional laws with
observations binned by two displayed features.

Vertical intervals in the gallery are empirical 10–90% prediction ranges
*within each bin*, caused by variations in omitted inputs. They are not
confidence intervals, estimator uncertainty, or identified sets.

Lattice of information and a monotonicity result
------------------------------------------------

Fix four binary feature measurements :math:`B_1,\ldots,B_4` and a fitted,
square-integrable prediction :math:`Y=f(X)`. Each feature subset
:math:`S\subseteq\{1,2,3,4\}` generates a sigma-algebra
:math:`\mathcal F_S=\sigma(B_j:j\in S)`. Subsets form the Boolean lattice
ordered by inclusion; adding a feature refines the partition of observations.

The average residual prediction variability is

.. math::

    R(S)=\mathbb E[\operatorname{Var}(Y\mid\mathcal F_S)]
       =\mathbb E[(Y-\mathbb E[Y\mid\mathcal F_S])^2].

**Proposition (monotonicity under refinement).** If :math:`S\subseteq T`,
then

.. math::

    R(S)-R(T)=\mathbb E\left[
       \left(\mathbb E[Y\mid\mathcal F_T]-
             \mathbb E[Y\mid\mathcal F_S]\right)^2\right]\ge0.

**Proof.** Conditional expectation is orthogonal projection in
:math:`L^2`. Since :math:`\mathcal F_S\subseteq\mathcal F_T`, their
projection subspaces are nested; the Pythagorean identity gives the result.

The example computes the corresponding empirical residuals on a *single
fixed held-out sample*, using fixed feature thresholds from training data.
As partitions refine, within-bin squared residuals cannot increase.
Its Shapley decomposition attributes the **reduction in unresolved
prediction variance**, not causal importance. Feature correlation and
choice of discretization affect the values.

Relationship to identification
------------------------------

A *different* set-valued map describes partial identification:

.. math::

    \mathcal I(u;\Gamma)=\{T(P):P\text{ compatible with the
    observations and restrictions at }u\}.

If larger :math:`\Gamma` weakens a constraint, the resulting identified
set can only expand. This is conceptually distinct from the conditional
prediction variation along a projection fiber.

Limits and validation
---------------------

- The reference distribution determines the measured model distances.
- The 2D embedding loses a quantified fraction of model variation.
- Exact neuron permutations imply identical functions globally; mere equality
  on the reference sample does not.
- Visual positions introduced for layout do not create model topology.
- More informative features reduce **average** residual variance, not
  necessarily uncertainty for every individual observation.
- A qualitative visual explanation alone is not evidence of improved human
  understanding. Benchmarking is required before methodological claims.

Related work
------------

Ainsworth, Hayase & Srinivasa, *Git Re-Basin*, ICLR 2023.
Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML 2019.
Klabunde et al., *Similarity of Neural Network Models*, ACM Computing Surveys 2025.
