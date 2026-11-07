# ModelAtlas (research prototype 0.5.0)

> **Research preview / manuscript development:** The SOM visualization uses fitted model parameters, but the photo-z partial-identification demo is synthetic and its bounds are conditional on explicit sensitivity assumptions. See [scientific notes](SCIENTIFIC_NOTES.md) and [reproducibility and manuscript checklist](REPRODUCIBILITY.md). The repository is public; the algorithms and scientific examples are under active research development and have not been released as validated calibration methods.

![ModelAtlas SOM with real input nodes and prototype connections](docs/figures/photoz_som_architecture.png)

*Matplotlib rendering of a fitted 9 × 9 SOM on synthetic galaxy photometry: five inputs, 81 codebook neurons, and 405 feature-to-prototype connections. The prism colors show an assumption-dependent identification-width metric, not the connection strengths.*


Compact, vector-native scientific model diagrams **derived from actual fitted parameters**, drawn as ordinary Matplotlib figures. ModelAtlas's goal is not to draw every model as if it were a neural network. It uses an *appropriate visual grammar* for each fitted model type.



## Symmetry-aware comparison of independent neural networks (research preview)

ModelAtlas now supports **exact hidden-unit sign/permutation gauge transformations** of
`scikit-learn` tanh binary MLPs and **data-dependent matching** between independent
models. In the latter case, training-sample neuron activation traces are aligned
with a Hungarian assignment; fitted predictions of both models are unchanged.

```bash
python -m pip install -e '.[examples,test]'
python examples/real_world_symmetry.py
```

The script reproduces real-data demonstrations on historical solar sunspots,
Wine chemistry, and Wisconsin Diagnostic Breast Cancer measurements, with
chronological train/test splitting for solar activity. It produces two-network
oblique diagrams, aligned residual atlases, numerical diagnostics and PDF/SVG/PNG
figures. **These tests do not establish model equivalence or explain causal
mechanisms.** Details: [Symmetry experiment notes](SYMMETRY_REAL_DATA.md).

## New experiment: adaptive prototype graph versus SOM and GTM (v0.5)

The package now has a **non-grid** Growing Neural Gas–style prototype-graph adapter,
a regularized Generative Topographic Mapping baseline, and vector-native
Matplotlib visualization of their learned topology and spectroscopic coverage.
The comparison uses **synthetic 8-band galaxy photometry**, no true redshift
labels for representation training, and nonrandom usable-spectroscopy selection.
See [Adaptive photo-z scientific notes](ADAPTIVE_PHOTOZ_NOTES.md) for
assumptions, precise figure semantics, controlled benchmarks and cautions.

```bash
python -m pip install -e '.[examples,test,adaptive]'
python examples/adaptive_photoz_comparison.py
python examples/adaptive_replicates.py
python -m pytest -q
```

The oblique graph image shows 8 photometric inputs, 64 photometric prototypes,
learned prototype edges and observed spectroscopic coverage. The rendered 3D
positions are PCA projections of the high-dimensional codebook, **not** an
intrinsic physical manifold. Faint input fibers encode selected codebook feature
components, **not neural-network weights or feature attribution**.

In five synthetic-mock repetitions, the adaptive graph improves photometric
quantization over the conventional SOM but does not improve redshift dispersion
or the assumption-dependent identified-set width. This is an instructive
negative result, not proof of superiority of either method.

## New in 0.3 — the **actual nodes and connections**

Previously, `plot_photoz_map` rendered cell diagnostics only. It did not draw
the SOM network. For a graph comparable to an input-feature → SOM-lattice
illustration, call `plot_photoz_architecture` or `visualize(..., architecture=True)`.

```python
from modelatlas import visualize

# `som` is the fitted 9x9 SOMMap; `bounds` is the PhotoZBounds result.
fig, ax = visualize(bounds, kind='photoz', som=som, architecture=True,
                    metric='identified_width', highlight=(7, 0))
fig.savefig('photoz_som_architecture.svg', transparent=True,
            bbox_inches='tight')
```

Alternatively, visualize the *fitted codebook* itself rather than an
identification metric:

```python
fig, ax = visualize(som, architecture=True, highlight=(7, 0))
```

- **Left discs:** one actual input feature per node (e.g., g-r, r-i, i-z, z-y, i).
- **405 faint colored strands:** each of the five features is connected to
  each of the 81 SOM neurons. Every link corresponds to a learned prototype
  component `som.codebook[row, column, feature]`, standardized before training.
  The *existence* of the connection is structural, not a learned variable.
- **Five heavier colored strands:** the five components of the highlighted
  prototype. Line width encodes absolute *relative standardized component*
  magnitude within each feature, not feature importance. Color indicates the
  input feature; the link drawing does not encode sign.
- **81 small dots:** SOM output neurons. Each tilted prism is one prototype
  in the imposed 2D lattice. Extruded 3D faces are a visualization choice.
- **Prism colors:** here, identified-set width for target
  $P_T(z>1.2\mid c)$ under the *stated* Wasserstein restriction. They are not
  network weights, redshifts, confidence intervals, or posterior uncertainty.
- **White outlined unit:** the cell with the largest *present contribution to
  the population bound width*, selected from observed data and assumptions;
  not a claim of optimal follow-up targeting or BMU of a single galaxy.

You can display fewer lines using `show_connections='selected'` or hide them
using `'none'` while retaining nodes and colored cell metrics.

Run `python examples/photoz_architecture.py` *after*
`python examples/photoz_identification.py`. The architecture example reads
the saved synthetic catalogue and the actual saved fitted codebook without
retraining the SOM or using hidden redshift labels.

**Interpretation warning:** SOM codebook coordinates are not causal
connections, generic neural-network layers, or scientifically meaningful
independent edges. Their values depend on feature scaling. If connections
are shown for a regression, GP, or another model family, a separate
model-specific adapter is needed.

## Demonstrated in this release

| Adapter | What the graphical marks actually mean | Current renderer |
|---|---|---|
| SOM with rectangular lattice | Each extruded tile is a real codebook vector; 2-D adjacency is the fitted SOM topology; color is the selected prototype feature, occupancy, or U-matrix | Compact oblique 3D-looking lattice |
| scikit-learn linear single-output regression | Signed fitted coefficients, in the scale actually used for fitting | Sparse coefficient stems |
| scikit-learn MLP (small network) | Actual nodes, learned signed weights and their magnitude; weaker arcs optionally omitted | Dense-layer network graph |

**Not implemented:** arbitrary PyTorch FX extraction, CNN, transformers, arbitrary R objects, hexagonal SOM lattices, interactive plots, hyperparameter uncertainty. The name and API are preliminary; this is not a published PyPI package.

## Installation

```bash
cd ModelAtlas
python -m pip install -e '.[examples,test]'
python examples/solar_activity.py
python -m pytest -q
```

Minimal runtime for plotting: `numpy`, `matplotlib`. `scikit-learn` and `pandas` are needed to run the complete optional solar activity demonstration; `statsmodels` is *not needed* to run it because the public-domain solar data are bundled.

## Working with a trained SOM

```python
import numpy as np
from modelatlas import SOMMap, visualize

# Codebook shape: (n_rows, n_columns, n_features), fitted feature scale.
codebook = np.load('som_codebook.npy')
calibration_features = np.load('scaled_training_vectors.npy')
view = SOMMap(codebook, samples=calibration_features,
              feature_labels=('log P', 'log R', 'T_eff'))

fig, ax = visualize(view, color_by=0, depth=.42,
                    azimuth=1.05, elevation='constant')
fig.savefig('som_map.svg', bbox_inches='tight', transparent=True)

# For scientific interpretation rather than only rendering:
print(view.diagnostics())
fig, ax = visualize(view, color_by='u_matrix')
fig.savefig('som_umatrix.pdf', bbox_inches='tight')
```

A trained MiniSom object can be passed directly: `visualize(minisom, color_by=0)`, or use `som_from_minisom(minisom, X=scaled_training_vectors)` if occupancy and fit diagnostics are wanted. The package's simple NumPy `train_som` is intended for demonstrators, not as a replacement for established training libraries such as TorchSOM or `kohonen`.

## Real astronomy example: solar-activity profiles

File: `data/sunspots_1700_2008.csv`. This is **real historical observational data**, not simulation. The source is the National Geophysical Data Center's yearly sunspot counts, 1700–2008, 309 observations, provided as the public-domain `statsmodels.datasets.sunspots` dataset. [Statsmodels documentation](https://www.statsmodels.org/v0.13.5/datasets/generated/sunspots.html).

A vector for each overlapping nine-year window uses the actual observations at years `t-4, t-2, t, t+2, t+4`. Thus **301 overlapping five-dimensional profiles**. All features are measured in the same units, so the example standardizes them using a single common mean and standard deviation; codebook values are inverted into the original sunspot scale for colored plotting. An 8×8 SOM is trained with a fixed seed. The map cannot be interpreted as a literal 3D physical solar manifold.

### Read the chart

- **One colored prism:** a model prototype, a learned time-profile with five real-valued components.
- **Hue:** the prototype's central annual sunspot count (in original data units). Color has physical meaning; decorative thickness normally does not.
- **Near each other in the grid:** neurons neighboring in the *SOM lattice*, which is an imposed rectangular topology. Their prototypes may or may not be similar; inspect the U-matrix.
- **Thin lines:** every input component is connected to every prototype component, as in an SOM's vector quantization architecture. They do **not** measure causality or feature importance.
- **Outlined prism:** best-matching unit of the maximum solar-activity year in the dataset.
- **Occupancy panel:** number of overlapping observational windows assigned to each unit. High occupancy can result from repeated historical patterns *and* from the fact that windows overlap.
- **U-matrix:** mean Euclidean distance to the four grid-neighbor prototypes, **in standardized feature units**. Large values indicate steep transitions across the map, but not automatically genuine separated physical clusters.
- **Quantization error:** mean Euclidean distance between each training profile and its best prototype in standardized coordinates.
- **Topographic error:** proportion of training profiles for which first and second nearest prototype units are not 4-connected neighbors on the rectangular lattice.

As a separate **predictive** experiment, the included Ridge regression predicts the following year using the preceding five annual values. It is trained on years up to 1920 and tested from 1930 onward (a gap avoids direct overlapping examples). In this particular fixed split, test $R^2$ is **0.805** for Ridge, **0.619** for a small MLP, and **0.612** for a last-value persistence baseline. These outcomes depend on the chosen split and hyperparameters; they are not a general comparison of model classes.

**Limitations:** contiguous windows are correlated; the distribution reflects observed history and measurement practice, not a selection-free population. SOMs offer an organized projection of high-dimensional data, not an identification theorem or a calibrated probability model. Feature scaling, initialization, SOM grid size and random seed can change the image. The sidewall extrusion is not a third independent learned dimension. When plotting another model, physical and statistical semantics must be remapped.

### Model-specific views

```python
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from modelatlas import visualize

ridge = Ridge().fit(X_train, y_train)
fig, ax = visualize(ridge, kind='linear', labels=feature_labels)

mlp = MLPRegressor(hidden_layer_sizes=(7,), random_state=1).fit(X_train, y_train)
fig, ax = visualize(mlp, kind='mlp', input_labels=feature_labels,
                    output_labels=['ŷ'])
```

For models with large layers, don't render millions of arbitrary individual edges: summarize groups/blocks, activation, attention or Jacobian metrics separately and clearly. Nonlinear models do not have a single globally valid linear coefficient plot.

## Longer-term architecture

`fitted model → adapter → typed scientific scene → Matplotlib or ggplot2 backend → SVG/PDF/PNG`.

Proposed stable scene primitives: `InputFeature`, `Parameter`, `Prototype`, `LatticeEdge`, `LearnedWeight`, `ComponentValue`, `ScalarDiagnostic`, and `Prediction`. Each has declared units, scale, transformation, aggregation, and uncertainty semantics. A generic node-edge-plot rendering with no adapter would be misleading. Add R `kohonen` and `torch` adapters, a `ggplot2` geom based on polygon vertices, then PyTorch `torch.fx` and probabilistic graphical-model adapters. Do not represent posterior uncertainties using geometric depth unless the mapping is explicitly defined.

## Output

The example creates `outputs/sunspot_som_oblique.{png,svg,pdf}`, U-matrix, occupancy, prototypes, fitted regression coefficient visualization, fitted MLP connections, a chronological predictive check for the linear model, and `model_diagnostics.json`. The scientific map and all diagnostics are generated by executable functions, not retouched images.

---

## New in v0.2: cellwise photo-z population-inference diagnostics

This release adds real code for photo-z partial-identification visualization, using an intentionally **synthetic** astronomical mock as the demonstration. The original SOM / neural-network / regression adapters are unchanged.

The observable fields are target photometry, a matched spectroscopic-redshift vector with `NaN` for unknowns, and an optional attempted-spectroscopy boolean mask. The hidden `true_z` column in the mock is stored separately and **never** used to compute the identified bounds.

### Run the complete example

```bash
python -m pip install -e '.[examples,test]'
python examples/photoz_identification.py
python -m pytest -q
```

Outputs are stored in `outputs/photoz/` with individually exportable vector `.svg` and `.pdf` counterparts to the `.png` previews. Figures include the same tilted 9x9 codebook geometry colored by target occupancy, successful spectroscopy coverage, observed high-redshift fraction, identified-set width, population-width contribution, spectroscopic attempts and conditional success; there is also a compact 2x2 diagnostic plot, a data-height relief variant, and a population sensitivity-interval plot.

### Use your own photometry/spectroscopy

```python
import numpy as np
from modelatlas import SOMMap, PhotoZCells, visualize, plot_photoz_diagnostics

# Fitted SOM codebook: rows x cols x n_photometric_features
# Features MUST be on the same fitted scaling as the SOM codebook.
som = SOMMap(fitted_codebook)

# X_target: (N,d) photometry for the actual parent/target catalogue
# z_spec: (N,) matched spectroscopic redshifts, NaN for *all* others
# attempted: (N,) booleans identifying which target objects were attempted
cells = PhotoZCells(som, X_target, z_spec, attempted=attempted)

# W1 radius Gamma in units of redshift, NOT spectroscopic success rate.
bounds = cells.identify(z0=1.2, gamma=0.14)
print(bounds.summary())
print(bounds.rank_cells(top=5))  # current bound-width contributions, not utility

fig, ax = visualize(bounds, kind='photoz', metric='identified_width')
fig.savefig('photoz_ambiguity.svg', bbox_inches='tight')

fig, axes = plot_photoz_diagnostics(bounds)
fig.savefig('photoz_four_planes.pdf', bbox_inches='tight')

# Optional: raise prisms using the metric as BOTH height and color.
fig, ax = visualize(bounds, kind='photoz', metric='identified_width',
                    elevation='metric')
```

The complete derivation, assumptions, interpretation, and limitations are in **`SCIENTIFIC_NOTES.md`**. In particular: (1) these are compatibility bounds conditional on an analyst-chosen sensitivity restriction, not confidence intervals; (2) an SOM makes a partition, it does not validate representativeness or estimate Gamma; (3) the attempted-target, failed-spectrum and successful-spectrum mechanisms should be modelled separately for a survey-grade analysis; (4) reporting population width contribution is not an expected follow-up information gain calculation.

The underlying code estimates sharp *relaxed probability-measure* tail bounds for a fixed empirical reference, allowing fractional transport of atoms. Do not claim sharp integer-valued finite-catalog bounds or finite-sample coverage without an additional layer of analysis.

---

## Experimental XAI: Explanation Atlas (v0.4 development)

![Observation-specific 3D explanation from the UCI Wine dataset](docs/figures/wine_class_one_explanation_atlas.png)

The oblique 3D model view now distinguishes **learned weights** from **local
path-integrated attributions**. Each input circle, edge and hidden neuron tile
encodes its signed contribution to the *change in target-class log-odds*
relative to a baseline. The same integration satisfies completeness at every
layer, and approximately conserves attribution across every hidden neuron.

The example uses real chemical measurements from the UCI Wine dataset, an
8–16–9–1 fitted scikit-learn classifier, and a standardized training-mean
baseline. It exports a contrasting observation so the two explanations can be
compared without changing the model or learned neuron arrangement.

```bash
python -m pip install -e '.[xai,test,examples]'
python examples/wine_explanation_atlas.py
python -m pytest -q
```

```python
import numpy as np
from modelatlas import explain_mlp, plot_explanation_atlas

# model: fitted binary sklearn MLPClassifier with two hidden layers
# X_train and x: already transformed with the training-fitted scaler
explanation = explain_mlp(model, x, baseline=np.zeros_like(x), steps=256)
fig, ax = plot_explanation_atlas(model, explanation,
                                 X_reference=X_train,
                                 feature_names=feature_names)
fig.savefig('explanation.svg', bbox_inches='tight')
print(explanation.max_layer_residual)
```

Scientific assumptions, exact attribution definition, limitations, numerical
checks and references are provided in [XAI_SCIENTIFIC_NOTES.md](XAI_SCIENTIFIC_NOTES.md).
**Research preview, not yet a validated explanation method for survey data.**