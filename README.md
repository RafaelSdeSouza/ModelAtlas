# ModelAtlas

**Created and scientifically led by Prof. Rafael S. de Souza.** Academic affiliations: Centre for Astrophysics Research, University of Hertfordshire (UK); Department of Physics and Astronomy, University of North Carolina at Chapel Hill (USA). Research connection: Universidade Federal do Rio Grande do Sul (UFRGS, Brazil), through the CNPq Conhecimento Brasil project.

ModelAtlas is an independent open-source research-software project; the academic affiliations above do not imply institutional endorsement. See the [project creator and credits](docs/guide/creator.rst), [citation guidance](docs/guide/citation.rst), and [CITATION.cff](CITATION.cff).

ModelAtlas makes Matplotlib figures from fitted scientific models. It draws
rectangular and hexagonal SOM codebooks, learned prototype graphs, small
neural networks, local logit attributions and symmetry-aligned comparisons.
Its statistical views show regression intervals and assumption-dependent
photo-z identified ranges.

**Research preview.** The solar, Wine, and Wisconsin examples use historical
or benchmark measurements. The photo-z examples use synthetic catalogues.
Their compatibility intervals condition on the observed reference and the
stated sensitivity restriction.

## Install from a checkout

```console
python -m pip install -e '.[examples,test]'
```

The Python distribution is ``modelatlas-viz``. Import ``modelatlas``.

```python
from modelatlas import fit_som, plot_som_architecture, save_figure

# x is a finite array of appropriately scaled fitted features.
som = fit_som(x, (7, 8), topology="hexagonal", seed=21)
fig, ax = plot_som_architecture(som, cmap="cividis")
save_figure(fig, "outputs/hexagonal_som")
```

## Build the documentation and figures

```console
python -m pip install -r requirements/tooling.txt
nox -s docs
```

Open ``docs/_build/html/index.html`` for the Sphinx gallery, quick start,
API reference, feature matrix, scientific interpretation, and data provenance.
Every gallery figure is rebuilt by running its Python example. The HTML
includes SVG, PDF and PNG downloads. The astronomy examples use two
bundled SDSS samples to map galaxy redshift and stellar temperature and
metallicity on hexagonal SOMs.

## Reproduce the original research examples

See the [adaptive photo-z notes](ADAPTIVE_PHOTOZ_NOTES.md),
[symmetry experiments](SYMMETRY_REAL_DATA.md),
[photo-z assumptions](SCIENTIFIC_NOTES.md) and
[attribution derivation](XAI_SCIENTIFIC_NOTES.md). The [original research walkthrough](https://github.com/RafaelSdeSouza/ModelAtlas/blob/5041936300e5660685b091e940796bd515326dd9/README.md) records the initial examples and results.

```console
MPLBACKEND=Agg python examples/solar_activity.py
MPLBACKEND=Agg python examples/photoz_identification.py
MPLBACKEND=Agg python examples/photoz_architecture.py
MPLBACKEND=Agg python examples/wine_explanation_atlas.py
MPLBACKEND=Agg python examples/real_world_symmetry.py
MPLBACKEND=Agg python examples/adaptive_photoz_comparison.py
MPLBACKEND=Agg python examples/adaptive_replicates.py
```

The architecture script consumes the synthetic catalogue and codebook from
the preceding photo-z script. The original findings and their caveats are
recorded in the linked notes. For a paper figure, cite the exact commit and
software version in [CITATION.cff](CITATION.cff).
