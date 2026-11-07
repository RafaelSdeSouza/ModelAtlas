Reproducibility and figure export
=================================

The gallery runs offline with Python 3.12, bundled survey subsets, the
datasets packaged with its dependencies, and a pinned figure environment.
From the repository root, reproduce the complete site and figures in an
isolated environment with one command after installing the pinned tooling.

.. code-block:: console

   python -m pip install -r requirements/tooling.txt
   nox -s docs

Open ``docs/_build/html/index.html``. The gallery has Python source and
notebook downloads, and the HTML bundle includes the complete set of
SVG, PDF and PNG figures under ``_static/generated/``. The pull request's
Documentation preview workflow also builds and archives the bundle.

To build directly in an active Python 3.12 environment:

.. code-block:: console

   python -m pip install -c requirements/figures-py312.txt -e '.[docs,examples]'
   MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 sphinx-build -W -n -b html docs docs/_build/html
   python scripts/check_docs.py docs/_build/html

The constraints are resolved for Linux x86-64. The inputs are in
``requirements/figures.in``. To refresh them after review:

.. code-block:: console

   uv pip compile requirements/figures.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 -o requirements/figures-py312.txt

Call :func:`modelatlas.save_figure` with a filename stem to write the three
formats. The package version is stored as creator or software metadata.
SVG IDs and timestamps are controlled. Reproduction across operating systems,
font installations or BLAS libraries may change fitted floating-point results
and the exact bytes. Record the commit SHA, package version, dataset version,
feature units, selection, training and evaluation split, seeds, model options,
and plotting scales with a manuscript figure.

Source datasets and earlier experiments
---------------------------------------

* **SDSS galaxies and SEGUE stars.** Two observational SOM studies use
  bundled, checksum-verified subsets. See :doc:`sdss_data` for citations,
  the exact sample cuts, and the FITS-to-NumPy recipe.
* **Solar activity.** Annual sunspot numbers from NOAA/NGDC for 1700–2008.
  The source is packaged with statsmodels and also archived in the original
  example's ``data/sunspots_1700_2008.csv``. Centered windows describe
  historical profiles. The original Ridge forecasting exercise trains on
  earlier years and tests on later years separated by a gap. The earlier
  symmetry experiment uses chronological holdout with overlapping windows.
* **Wine.** UCI Wine chemical measurements from ``sklearn.datasets.load_wine``.
  The attribution and alignment examples label class zero versus the rest.
  Scaling and layout for local explanations use only training observations.
* **Wisconsin.** UCI Wisconsin Diagnostic Breast Cancer measurements from
  ``sklearn.datasets.load_breast_cancer``, with a stratified split. The
  experiment illustrates aligned representations on one split.
* **Synthetic photo-z.** Fixed-seed simulated photometry and spectroscopic
  selection. The original multi-seed experiment uses toy bandpasses. Truth
  redshifts enter audit diagnostics after fitting and identification.

The original commands remain available from a checkout or source
distribution. Use the pinned figure environment to compare reruns.

.. code-block:: console

   python examples/solar_activity.py
   python examples/photoz_identification.py
   python examples/photoz_architecture.py
   python examples/wine_explanation_atlas.py
   python examples/real_world_symmetry.py
   python examples/adaptive_photoz_comparison.py
   python examples/adaptive_replicates.py

Run the two photo-z scripts in the stated order. The second consumes the
first script's saved mock and codebook. The original outputs go to
``outputs/`` and ``docs/figures/``, with the adaptive summaries beside
``docs/figures/``. The archived five-seed comparison reports a modest
adaptive-graph quantization gain without a corresponding improvement in
target redshift dispersion or identified width. Refer to the original
`adaptive photo-z notes <https://github.com/RafaelSdeSouza/ModelAtlas/blob/research/explanation-atlas-v0.4/ADAPTIVE_PHOTOZ_NOTES.md>`__
for the controlled experiment and the assumption violations.
