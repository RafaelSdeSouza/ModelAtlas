# Scientific semantics: SOM cellwise photo-z partial identification

This is a **research prototype** whose population-compatibility bounds are not statistical confidence regions, credible intervals, prediction bands, or guarantees under unknown selection bias. It does not replace point estimation or spectroscopic-calibration methods.

## Observations and target

A target catalogue contains `N` objects with photometric features `X_i`. An SOM is fitted to this target photometry (using no redshifts) and defines a partition `C_i = c(X_i)`. For a subset, a matched usable spectroscopic redshift `Z_i` is available. Every target belongs to exactly one cell.

Within cell `c`, let `N_c` be the count of target observations and `S_c` the count with a usable measured redshift. Define `p_c = S_c / N_c`. The measured redshift distribution is the fixed empirical probability measure `P_S,c`, and the missing redshift distribution is unknown `P_M,c`. The target redshift distribution in the cell is

`P_T,c = p_c P_S,c + (1-p_c) P_M,c`.

We focus on the high-redshift tail `theta_c=P_T,c(Z > z0)`, and globally `Theta=sum_c (N_c/N) theta_c`.

### Sharp bounds without assumptions about missing redshifts

For cell `c` with usable spectra:

`p_c * theta_S,c <= theta_c <= p_c*theta_S,c + (1-p_c)`.

If `S_c=0`, the interval for the populated cell is `[0,1]`. This is an *identified* range from known counts and redshifts; the unobserved redshifts can lie anywhere on the redshift support. It is not a CI.

### Optional distributional sensitivity model

For each cell with at least one usable spectroscopic redshift, assume

`W1(P_M,c, P_S,c) <= Gamma`,

where `W1` uses absolute redshift distance, so `Gamma` is expressed in redshift units. `Gamma=0` enforces exact within-cell transfer. `Gamma=None` drops the transfer restriction. `Gamma>0` allows the unobserved conditional redshift distribution to differ from the observed one. **Gamma is not estimated from usable spectra alone**, and its units should not be confused with a spectroscopic success-probability parameter sometimes also denoted gamma.

The code computes the sharp infimum and supremum of `P_M,c(Z > z0)` relative to the empirical `P_S,c` by moving the cheapest mass across the threshold in redshift space first, paying its minimum transport cost. Since the reference is discrete, individual reference atoms are allowed to split. For atoms exactly at the strict threshold, the extremum can be approached without being attained for arbitrarily small positive Gamma. At Gamma=0 we return exact transfer. These are **sharp for the relaxed probability-measure model treating the empirical successful-spectrum distribution and cell mixing fractions as fixed**, not necessarily attainable as exact integer outcomes in a finite unobserved catalogue.

Write `a_c(Gamma), b_c(Gamma)` for those tail extremes. Then

`L_c = p_c theta_S,c + (1-p_c) a_c(Gamma)`

`U_c = p_c theta_S,c + (1-p_c) b_c(Gamma)`

and, with separate constraints across cells, the population bounds are

`[sum_c (N_c/N)L_c, sum_c(N_c/N)U_c]`.

The total interval width decomposes additively into `sum_c (N_c/N)(U_c-L_c)`; this is precisely the `population_width_contribution` map. A cell with a large width but very few target objects can contribute little to population-level ambiguity. A cell with *no* usable spectrum cannot inherit a Wasserstein reference from its own cell and retains `[0,1]` in this implementation.

## What the metrics encode

| Map | Numerator / denominator | Interpretation |
|---|---|---|
| `target_occupancy` | `N_c` | Target photometric density projected into the SOM |
| `spectroscopic_coverage` | `S_c/N_c` | Fraction with a matched usable spectral redshift |
| `attempt_fraction` | `attempted_c/N_c` | Fraction selected for a spectroscopy attempt |
| `success_given_attempt` | `S_c/attempted_c` | Conditional redshift-measurement success |
| `observed_high_fraction` | `number of usable spectra with Z>z0 / S_c` | Spectroscopic subsample's point fraction; not a target population estimate without transfer assumptions |
| `identified_width` | `U_c - L_c` | Compatibility range for the target high-z fraction in that cell |
| `population_width_contribution` | `(N_c/N)*(U_c-L_c)` | Additive share of total current population bound width |

In the oblique plot, the 2D grid topology is imposed by the SOM lattice; it is not a measured 2D/3D physical manifold. In constant elevation mode, prism thickness is cosmetic. In metric elevation mode, *both height and hue encode the named scalar cell metric*; no new latent dimension is implied. Every exported chart is an ordinary Matplotlib Figure/Axes and uses vector paths in SVG/PDF output.

## Synthetic experiment

`examples/photoz_identification.py` generates an approximate five-band toy photometric catalogue using a 4000Å break, galaxy SED slope/dust, emission lines, broad passbands, magnitude-dependent noise and a multi-component redshift distribution. Both spectroscopy targeting and redshift success are non-random. The SOM uses only the full target photometry. Identification calculations read only the available spectroscopic redshifts. A *separate*, explicitly named ground-truth file is generated for simulation validation; do not use hidden truth to compute bounds.

The experiment is not DESI, HSC, COSMOS, LSST or a calibrated mock from those surveys. Do not treat the instrument passbands or selection functions as survey-calibrated. Any manuscript application needs actual redshift/photometry matching, footprint support, selection-audit diagnostics, uncertainty in observed redshift distributions, and defensible Gamma sensitivity analyses.

## Finite-sample and model validation caveats

- These intervals condition on the observed spectroscopy distribution as though fixed. Finite spectroscopy sample noise, catastrophic redshift errors, redshift dependence of cell assignment, and photometric uncertainty are **not included**.
- The assumption may fail even when the hidden truth happens to land inside the reported interval. The mock audit *can* inspect the hidden full redshifts to demonstrate that issue; real observations cannot.
- The current `Gamma` is an analyst-specified sensitivity restriction; the identification code cannot learn it from the successful sample alone.
- Empty spectra within a cell are handled conservatively. Limited support/selection bias does not disappear because a SOM has been fitted.
- The current width ranking is not the expected value of another redshift observation: follow-up design additionally needs acquisition cost, outcome-dependent success rates, attainable sampling designs, and a decision-theoretic objective.

## Formal extension candidates

1. Reconcile finite-sample confidence sets with identified sets, reporting both separately.
2. Separate (i) unattempted, (ii) attempted but failed, and (iii) usable redshifts as different selection mechanisms, each with calibrated sensitivity restrictions.
3. Admit target/spectroscopic footprint mismatch, partial overlap, and photometric support violations.
4. Add joint constraints across cells (rather than independent W1 budgets) and optimize functional bounds via a global linear/convex program.
5. Offer a more general partition adapter for SOM, k-means, cluster ensembles, and learned neural embeddings, maintaining exactly the same inferential semantics.
6. Build a ggplot2 backend from vector-scene primitives, retaining Matplotlib as the reference rendering engine.
