"""
Stellar temperature and metallicity in SEGUE
==============================================

How do effective temperature and iron abundance vary across the same
stellar color map? We use two dereddened SDSS colors for a sample of
F- and G-like dwarfs. `Ivezić et al. (2008) <https://arxiv.org/abs/0804.3850>`__
used these colors to study stellar metallicity. Here a hexagonal SOM
gives two aligned views of how spectroscopic estimates vary across
the observed color locus.

The :download:`catalogue subset <sdss_dr9_stars.npz>` contains 8,000
stars with SDSS DR9 SEGUE Stellar Parameter Pipeline estimates. Put
it beside the Python script or notebook to rerun this example. The
data source, quality cuts, and reddening correction are in
:doc:`../guide/sdss_data`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from modelatlas import ObliqueProjection, fit_som, plot_cell_plane

directory = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
with np.load(directory / "sdss_dr9_stars.npz", allow_pickle=False) as catalogue:
    colors = catalogue["colors"]
    temperature = catalogue["teff"]
    metallicity = catalogue["feh"]
    plates = catalogue["plate"]

# %%
# Fit one map and use it twice
# ----------------------------
# Fit the scaler and SOM on training-plate photometry. Assign stars
# from the held-out plates to the fitted prototypes. For cells with
# at least 12 held-out spectra, compute a median for each pipeline
# quantity. Both maps use the same cell locations.

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=29)
train, test = next(split.split(colors, groups=plates))
assert not np.intersect1d(plates[train], plates[test]).size
scaler = StandardScaler().fit(colors[train])
som = fit_som(
    scaler.transform(colors[train]),
    (8, 10),
    topology="hexagonal",
    steps=10_000,
    seed=31,
    feature_labels=(r"$(u-g)_0$", r"$(g-r)_0$"),
)
shape = som.lattice.shape
cells = np.ravel_multi_index(som.best_matching_units(scaler.transform(colors[test])).T, shape)
counts = np.bincount(cells, minlength=np.prod(shape))
supported = np.flatnonzero(counts >= 12)
median_temperature = np.full(counts.size, np.nan)
median_metallicity = np.full(counts.size, np.nan)
for cell in supported:
    sample = test[cells == cell]
    median_temperature[cell] = np.median(temperature[sample])
    median_metallicity[cell] = np.median(metallicity[sample])

camera = ObliqueProjection(azimuth=0, elevation=90)
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.9), layout="constrained")
for ax, title, values, unit, cmap in zip(
    axes,
    ("Effective temperature", "Iron abundance"),
    (median_temperature, median_metallicity),
    ("Median temperature (K)", r"Median $[\mathrm{Fe/H}]$ (dex)"),
    ("cividis", "viridis"),
):
    plot_cell_plane(
        values.reshape(shape),
        label=unit,
        cmap=cmap,
        ax=ax,
        topology=som.topology,
        projection=camera,
    )
    ax.set_title(title, loc="left", fontsize=11, pad=9)
fig.set_label(
    "Two aligned hexagonal maps of stellar colors. The left map shows "
    "median effective temperature and the right map shows median iron abundance. "
    "Gray cells contain fewer than 12 held-out spectra."
)
print(f"{len(train):,} training stars and {len(test):,} held-out stars.")
print(f"{len(supported)} of {counts.size} cells contain at least 12 held-out spectra.")

# %%
# Locate the prototypes in the measured colors
# --------------------------------------------
# Convert the prototype coordinates back to magnitudes. Each colored
# circle is a supported cell at its prototype coordinates.
# The small grey points show individual held-out stars.

prototypes = scaler.inverse_transform(som.codebook.reshape(-1, 2))[supported]
fig, ax = plt.subplots(figsize=(6.25, 4.2), layout="constrained")
ax.scatter(colors[test, 1], colors[test, 0], s=5, c="#8d98a2", alpha=0.24, linewidths=0)
points = ax.scatter(
    prototypes[:, 1],
    prototypes[:, 0],
    s=48,
    c=median_metallicity[supported],
    cmap="viridis",
    edgecolors="#172c3c",
    linewidths=0.45,
)
fig.colorbar(points, ax=ax, label=r"Median $[\mathrm{Fe/H}]$ (dex)")
ax.set(xlabel=r"Dereddened $(g-r)_0$ (mag)", ylabel=r"Dereddened $(u-g)_0$ (mag)")
ax.spines[["top", "right"]].set_visible(False)
fig.set_label(
    "The stellar color locus in held-out SDSS photometry. Colored circles "
    "locate the SOM prototypes, with hue showing their median iron abundance."
)

# %%
# Read the maps within the selected sample
# ----------------------------------------
# The two maps summarize how the SSPP estimates vary across the color
# locus. Targeting, reddening corrections, and the pipeline's parameter
# uncertainties affect these patterns. The medians describe this
# spectroscopic sample. Application to another stellar population would
# require its own selection and calibration analysis.
