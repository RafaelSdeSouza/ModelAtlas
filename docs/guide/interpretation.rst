Scientific interpretation
=========================

Lattice geometry and projection
-------------------------------

A SOM prototype is a vector in fitted feature coordinates. Its displayed
location is the index of that vector in the imposed lattice. In an odd-row
hexagonal lattice, a cell at row :math:`r` and column :math:`c` has the
planar center

.. math::

   (x, y) = \left(c + \tfrac{1}{2}(r \bmod 2),\,\tfrac{\sqrt{3}}{2}r\right).

Each interior hexagon has six neighbors at unit Euclidean distance.
With axial coordinates :math:`(q,r) = (c - \lfloor r/2 \rfloor, r)`,
the hop distance is :math:`\max(|\Delta q|,|\Delta r|,|\Delta q+\Delta r|)`.
Hexagonal online updates apply a Gaussian to this hop distance. The
rectangular trainer retains its Gaussian in Euclidean grid coordinates
and uses four-connected neighbors for the U-matrix and topographic error.
A topographic error is the fraction of observations whose first two
best-matching units are separated by more than one lattice edge. All
feature-space distances depend on the chosen preprocessing.

The prism camera maps world coordinates to two orthonormal screen axes.
For azimuth :math:`a` measured from positive y toward positive x, and
elevation :math:`e` above the xy plane, the rows of its matrix are

.. math::

   u=(\cos a, -\sin a, 0),\qquad
   v=(-\sin e\sin a,-\sin e\cos a,\cos e).

The camera collapses the direction :math:`v \times u`. A unit ground area
projects to :math:`\sin e` screen area before resizing. Projection preserves
incidence and parallel lines and foreshortens off-screen distances. Prism
thickness is normally constant in lattice units. Occupancy and scalar-relief
modes explicitly specify how they map a value to thickness. Compare
component planes or the U-matrix for quantities the apparent 3D shape can
obscure. Nearby caps can overlap at shallow elevations or with tall relief.

A graph has a different geometry. Edges come from learned adjacency.
Cap centers are projected from up to three unwhitened principal axes of the
fitted prototypes. The decorative screen-space extrusion adds depth to the
glyph. Width is :math:`0.13 + 0.17\sqrt{n/n_{\max}}` in normalized display
units. A nonzero width makes an unoccupied prototype visible. Hue is
either the count or the usable-spectroscopy fraction, according to the
colorbar. Gray caps have no assigned targets. The fine strands show
centered codebook components with per-feature scale and magnitude clipping,
for a subset ordered by occupancy.

The layout reports retained PCA variance and the distance stress
:math:`\lVert d-s\tilde d\rVert_2/\lVert d\rVert_2`, where :math:`d`
and :math:`\tilde d` are distances for the same pairs in fitted space and on
screen. The scalar :math:`s` minimizes this residual. The former statistic
summarizes the PCA stage, while stress and pairwise correlation assess the
full mapping. Pooled fit coordinates permit visual comparison across graphs
with identical feature preprocessing. Close eigenvalues can make the
projection unstable under perturbations even with a fixed sign convention.

Weights, local attributions, and model symmetry
-----------------------------------------------

In an MLP weight diagram, an edge belongs to a fitted matrix. The
structural view scales and thresholds magnitudes within layers. Weight
sign and magnitude are parameter properties. Interpretation can change
after a symmetry transformation.

For a binary MLP with output logit :math:`f`, take a baseline :math:`b`
and a straight path :math:`x_t=b+t(x-b)`. Let :math:`a_{\ell-1,j}`
denote an incoming activation and :math:`z_{\ell,k}` the next preactivation.
The contribution of an edge is

.. math::

   E_{\ell,jk}=\int_0^1
   \frac{\partial f}{\partial z_{\ell,k}}(x_t)
   W_{\ell,jk}\frac{d a_{\ell-1,j}(x_t)}{dt}\,dt.

By the chain rule the sum at each layer equals :math:`f(x)-f(b)`. Input
sums give Integrated Gradients. Incoming and outgoing sums at the same
hidden unit agree up to numerical error. Check
``max_layer_residual`` and ``max_node_residual``, especially near ReLU
kinks. The probability displayed at the output is a separate annotation.
Attributions depend on the baseline and the path. The highlighted trace
selects large local edge terms and has no causal interpretation. Layout
uses an approximate assignment of reference activation signatures to grid
slots, and fixed scales are necessary to compare color strength across cases.

If hidden activations are tanh, permutations and sign changes of hidden
units preserve the fitted function. With row-vector activations and signed
permutation matrices :math:`S_\ell`, transform layer weights and biases as

.. math::

   W'_\ell=S_{\ell-1}^{\mathsf T}W_\ell S_\ell,\qquad
   b'_\ell=b_\ell S_\ell,\qquad S_0=S_L=I.

Oddness of tanh and orthogonality of the signed permutations give
:math:`a'_\ell=a_\ell S_\ell`. The input ordering and output logit are
preserved. Compare independent fits by solving a signed assignment on
training activation traces. Duplicate or inactive units can have multiple
equally good matches. Inspect held-out activation residuals and prediction
agreement separately. Variability between two fitted models describes that
pair of fits. A sampling interval requires a separate inferential procedure.

Uncertainty and partial identification
--------------------------------------

The regression plotters accept interval endpoints computed elsewhere. A
coefficient confidence interval concerns parameter estimation. A prediction
interval concerns a future response under the fitted assumptions. Their
interpretations belong in the axis and legend labels.

Usable spectroscopy divided by target counts measures coverage. The
attempt fraction and conditional success require recorded attempts. The
plots mark those statistics as missing when attempt records are unavailable.

A photo-z identified interval describes the values compatible with the
observable data and the assumed restrictions. For a populated cell
:math:`c`, write :math:`q_c` for the fraction with usable spectroscopy,
:math:`F_{S,c}` for its empirical redshift distribution, and :math:`F_{M,c}`
for the unknown distribution of all remaining targets. The target
distribution is

.. math::

   F_{T,c}=q_cF_{S,c}+(1-q_c)F_{M,c}.

For :math:`\theta_c=P(Z>z_0\mid c)`, unrestricted missing redshifts give
the range from :math:`q_c\theta_{S,c}` to
:math:`q_c\theta_{S,c}+1-q_c`. An optional assumption is
:math:`W_1(F_{M,c}, F_{S,c})\leq\gamma`, where :math:`\gamma` has
redshift units. At zero radius and positive coverage the missing
distribution equals the empirical reference. At any positive radius,
the tail extrema follow by moving the cheapest empirical mass across the
threshold first. The strict upper tail can have a supremum that is only
approached. With no usable spectra, the populated cell retains :math:`[0,1]`.

The code treats the observed fraction, the target cell weights and the
empirical redshift distribution as fixed. The bounds are sharp for relaxed
probability measures with fractional atom transport and independent cell
restrictions. Finite-catalog integer completions can give different
endpoints. Sampling uncertainty in the reference and the validity of the
radius require separate analyses. Cell weights sum the interval endpoints
and provide a width decomposition. Follow-up utility additionally depends
on targeting, success probabilities, acquisition costs and how new data
would change the restrictions.

All photo-z demonstrations in this repository use synthetic catalogues.
The examples illustrate the consequences of assumptions under a specified
simulation. A survey analysis would require independently sourced photometry
and spectroscopy, documented target and footprint matching, and external
validation of the sensitivity assumptions.
