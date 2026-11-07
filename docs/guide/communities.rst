Research communities and workflows
==================================

ModelAtlas is a model-agnostic scientific visualization library, with
**astronomy as its primary documentation domain**. The examples are
chosen to reflect the questions faced by research communities using
machine learning, self-organizing maps (SOMs), probabilistic models,
and interpretable AI. These examples do not imply endorsement by
any survey or collaboration.

Galaxy surveys and photometric redshifts
----------------------------------------

**Rubin/LSST DESC, Euclid, DES, and spectroscopic follow-up teams** are
natural audiences for prototype maps and predictive model comparison.
Browse :doc:`../astronomy/plot_sdss_galaxy_som`,
:doc:`../gallery/plot_functional_lattice`, and
:doc:`../gallery/plot_probability_geometry`.

Use the SOM visualizations to explore photometric neighborhoods and to
display where spectra exist. Use the functional atlas to compare ML models
on the **same** reference photometry. A comparison using only a selected
spectroscopic catalogue cannot establish the redshift distribution of a
different photometric parent sample. The
:doc:`../gallery/plot_photoz_bounds` example demonstrates this limitation
on synthetic data under explicit sensitivity restrictions.

Stellar populations and spectroscopic labels
---------------------------------------------

**Gaia, SDSS/SEGUE, stellar-parameter pipelines, and population studies**
use supervised estimators, nonlinear representations, and prototype maps
to relate photometry and spectra to stellar properties.

The :doc:`../astronomy/plot_sdss_stellar_som` tutorial holds neuron
positions constant while coloring them by effective temperature and
metallicity. The :doc:`../gallery/plot_prototype_graph` tutorial
contrasts fixed latent geometry with learned graph connectivity on
the same measured colors.

Time-domain astronomy and survey alerts
----------------------------------------

**Rubin/LSST alert brokers, ALeRCE, Fink, and related time-domain
classification communities** use ML to score, classify, prioritize, and
monitor changing sources. The :doc:`../gallery/plot_aligned_networks`
example tests a real, chronologically held-out solar-activity prediction
task, illustrating why aligning hidden neurons does not imply equivalent
predictions. Solar annual activity is **not** a transient alert stream:
a reproducible astronomical-alert benchmark is a future development.

Explainable astronomical machine learning
------------------------------------------

A neural network's hidden-unit labels can change without changing its
function. Visualizations should distinguish trained weights, integrated
attributions, observation-specific activations, and
symmetry-aligned residual differences. The
:doc:`../gallery/plot_local_explanations` tutorial applies those ideas
to galaxies with real SDSS colors and redshifts.

The general research question is whether model visualization improves
the interpretation of fitted science models, while making geometric
distortion, missing reference coverage, and incomplete identification
explicit. The present gallery is **illustrative**; formal user-study
validation of interpretability remains future work.

Implementation and data provenance
----------------------------------

All gallery images come from Python/Matplotlib rather than generative
illustrations. The SDSS examples use bundled, documented subsets with
independent plate-group splits for model fitting and evaluation.
See :doc:`sdss_data` and :doc:`reproducibility` for provenance and
one-command regeneration.
