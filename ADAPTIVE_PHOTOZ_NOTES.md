# Adaptive representations for photo-z — controlled research preview

**Version 0.5 experiment; exploratory, not a performance claim about a survey.**

This module compares three unsupervised geometric representations of the **same
8-dimensional target photometry**:

1. **SOM**: 8 × 8 rectangular map fitted with the package's online trainer.
2. **GTM**: 8 × 8 latent-grid *regularized* Generative Topographic Mapping
   (radial-basis mapping, isotropic residual, EM; **MAP hard assignments used
   for cellwise comparison**, sacrificing GTM's native soft memberships).
3. **GNG-style prototype graph**: incremental Growing Neural Gas with adaptive
   units and competitive-Hebbian learned graph edges. The compact prototype
   reconnects isolated units instead of deleting them, deviating from the
   canonical Fritzke procedure. Treat it as a baseline to validate against a
   vetted implementation before using it in a paper.

The mock uses eight approximate band-integrated SED fluxes (`u,g,r,i,z,y,J,H`),
galaxy-type and dust variation, broad breaks, emission features,
heteroscedastic noise, and a synthetic apparent magnitude. It uses **toy
Gaussian bandpasses, not LSST/HSC throughput curves**. Colors and a magnitude
produce eight training inputs, standardized using training photometry only.
Targeting depends on magnitude and colors; spectroscopic success additionally
depends on true redshift and simulated type. These dependencies are known to the
simulator but unavailable to the identification code.

## Reproduce

```bash
python -m pip install -e '.[examples,test,adaptive]'
python examples/adaptive_photoz_comparison.py
python examples/adaptive_replicates.py
python -m pytest -q
```

Outputs include `docs/adaptive_metrics.json`, `docs/adaptive_replicates.csv`,
`docs/adaptive_replicates_summary.json` and vector/PDF figures under
`docs/figures/adaptive_*`. Runs use explicit seeds and a fixed 64-unit budget.
The first 4,500 photometric objects fit the representations; 1,500 held-out
objects measure photometric quantization; **all 6,000 target objects** are
assigned to fitted prototypes for selection/identification diagnostics.
The redshift truth is used **only for audit diagnostics**, never for training
or for calculating identified bounds.

## Single-realization diagnostics (seed 39)

| Quantity | SOM | GTM | GNG-style |
|---|---:|---:|---:|
| Prototypes | 64 | 64 | 64 |
| Learned/displayed graph edges | 112 (grid) | 112 (latent grid) | 165 (learned) |
| Held-out photometry mean nearest-prototype distance | 1.586 | 1.549 | 1.498 |
| Target-weighted within-cell SD of **true** z (simulation audit) | 0.184 | 0.193 | 0.190 |
| Identified width for target P(z>1.2), Γ=0.16 | 0.387 | 0.395 | 0.390 |
| Target-weighted fraction violating W1≤0.16 among comparable cells (simulation audit) | 10.6% | 15.0% | 11.6% |

The full target tail fraction is **38.17%**, whereas the successful
spectroscopic subsample has **21.53%**. More accurate photometric quantization
does not imply narrower or better-justified identification intervals.

## Five independent simulations (seeds 39, 59, 79, 99, 119)

Means (sample SDs across simulations) from `docs/adaptive_replicates.csv`:

| Measure | SOM | GTM | GNG-style |
|---|---:|---:|---:|
| Held-out quantization distance | 1.574 (0.020) | 1.538 (0.023) | **1.491 (0.017)** |
| Mean within-cell true-z SD (audit only) | **0.193 (0.008)** | 0.200 (0.006) | 0.195 (0.004) |
| Population identified-width Γ=0.16 | **0.388 (0.008)** | 0.398 (0.014) | 0.396 (0.011) |
| Population fraction violating W1 restriction (audit only) | 0.104 (0.036) | 0.125 (0.022) | 0.158 (0.028) |

All five simulations happened to have the target fraction inside each
reported plug-in interval. **That does not validate the W1 assumption**;
indeed, it is violated in some cells for every method. The intervals are
population-compatibility ranges conditional on the proposed assumptions and
empirical successful-spectra distributions, not confidence intervals or
coverage-calibrated statistical procedures.

## Statistical meaning of the bounds

For a cell `k`, let `q_k = N_success,k / N_target,k` and `F_S,k` be the
empirical distribution of usable spectroscopic redshifts. The unknown
missing-spectra distribution `F_M,k` is restricted by

    W_1(F_M,k, F_S,k) <= Gamma

with `Gamma=0.16` in redshift units. For
`theta_k=P_target(Z>1.2 | k)`, the bounding code computes the infimum and
supremum over this Wasserstein ball, multiplied by the observed mixing counts.
A populated cell with no usable spectrum retains interval [0,1]. Cell bounds
are combined using **target occupancy weights**, assuming independent
cellwise sensitivity restrictions. Target photo-z inference with these bounds
does not become identified merely because an embedding is more flexible.

`Gamma=0.16` is chosen for demonstration, **not estimated or guaranteed**.
The simulation audit measures the true W1 between successful and missing
redshift distributions in each cell; its violations invalidate the restrictive
assumption in those cells. A paper needs explicit sensitivity/robustness
analysis, finite-sample uncertainty, and justified target/spectroscopy matching.

## Figure semantics

- **3D oblique graph**: each small prism is one learned photometric prototype,
  its color the observed *usable-spectroscopy fraction*, and its width a
  monotone function of target occupancy. The tiny dot marks the neuron.
- **Graph edges**: actual competitive-Hebbian edges learned from photometric
  vectors (not arbitrary grid neighbors). Edges are not causal relationships.
- **Eight input discs**: u−g, g−r, r−i, i−z, z−y, y−J, J−H and i magnitude.
  Faint blue/orange fibers are *selected, signed, standardized prototype
  feature coordinates*. They are **not MLP connection weights**; not every
  possible input-to-prototype strand is drawn.
- **3D arrangement**: projection of the **first three principal components of
  the 8D learned codebook** and a fixed oblique camera. The first three PCs
  explain **68.5%** of prototype variation for seed 39. Prism extrusion is
  decorative, not a fourth latent feature or redshift axis.
- **Three-model comparison**: shared PCA space of the standardized
  photometric observations, allowing direct comparison of the codebook
  layouts. GTM lattice edges indicate latent-grid neighbors; its point
  positions are generated prototypes, not input observations.
- **Topographic error** in the diagnostic JSON is the fraction for which
  two closest prototypes are not directly adjacent in the method's graph.
  **This is not a fair direct cross-method performance metric when edge
  budgets differ** (112/112/165 edges in the showcased realization).

## Research conclusion and next hypotheses

These tests falsify an attractive but overly simple claim: that the adaptive
prototype graph must outperform an SOM for photo-z calibration. It improves
vector quantization on the mock, but does not consistently reduce true
redshift mixture complexity or the identified-set width.

Potential *new* objectives to test against SOM include (i) probabilistic
and uncertainty-aware prototypes trained on the full photometric population;
(ii) redshift-blind reconstruction plus conditional representation audits;
(iii) explicitly measuring within-cell degeneracy and identifying the cells
where spectroscopy would be informative; and (iv) learning partitions that
minimize an observable *bound-width objective* without leaking the missing z.
Such objectives must be assessed against data leakage and additional
assumptions. Without spectroscopy or extra information, two photometrically
indistinguishable redshift populations need not be point identified.

## Literature and attribution

- Fritzke (1995), *A Growing Neural Gas Network Learns Topologies*, NeurIPS 7.
  https://papers.nips.cc/paper/1994/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html
- Bishop, Svensén & Williams (1998), *GTM: The Generative Topographic Mapping*,
  Neural Computation 10:215–234.
  https://www.microsoft.com/en-us/research/publication/gtm-the-generative-topographic-mapping/
- DES Y6 photo-z calibration (2026), including clustering redshifts to address
  SOM color–redshift degeneracies. https://doi.org/10.1103/6lfl-h6dg