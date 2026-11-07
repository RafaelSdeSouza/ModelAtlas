"""
Galaxy colors and spectroscopic redshift in SDSS
==================================================

How much do four broadband colors tell us about a galaxy's redshift? We
fit a hexagonal self-organizing map to observed SDSS DR8 galaxy colors,
then read the median and the distribution of spectroscopic redshifts in
the cells on held-out plates. The workflow follows the idea of mapping
the empirical color–redshift relation in
`Masters et al. (2015) <https://arxiv.org/abs/1509.03318>`__.

The :download:`catalogue subset <sdss_dr8_galaxies.npz>` contains 12,000
bright spectroscopic galaxies. Put it beside the Python script or notebook
to rerun this example. See :doc:`../guide/sdss_data` for the catalogue
selection and source files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from modelatlas import ObliqueProjection, fit_som, plot_cell_plane, plot_som_architecture

directory = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
with np.load(directory / "sdss_dr8_galaxies.npz", allow_pickle=False) as catalogue:
    colors = catalogue["colors"]
    redshift = catalogue["redshift"]
    plates = catalogue["plate"]

# %%
# Fit on photometry from training plates
# --------------------------------------
# The split assigns entire spectroscopic plates to either training or
# evaluation. Scaling uses only the training colors. Each prototype
# represents four differences between observed model magnitudes.
# Spectroscopic redshift enters the analysis after the map is fitted.

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
train, test = next(split.split(colors, groups=plates))
assert not np.intersect1d(plates[train], plates[test]).size

scaler = StandardScaler().fit(colors[train])
som = fit_som(
    scaler.transform(colors[train]),
    (9, 11),
    topology="hexagonal",
    steps=14_000,
    seed=26,
    feature_labels=(r"$u-g$", r"$g-r$", r"$r-i$", r"$i-z$"),
)
shape = som.lattice.shape
cells = np.ravel_multi_index(som.best_matching_units(scaler.transform(colors[test])).T, shape)
z_test = redshift[test]
counts = np.bincount(cells, minlength=np.prod(shape))
supported = np.flatnonzero(counts >= 12)
medians = np.full(counts.size, np.nan)
for cell in supported:
    medians[cell] = np.median(z_test[cells == cell])

# %%
# Read the color–redshift map
# ---------------------------
# Each hexagon is one learned color prototype. Color shows the median
# spectroscopic redshift of galaxies assigned to that cell on held-out
# plates. Grey hatching marks cells with fewer than 12 such galaxies.
# The strands identify the input features. Prism depth is decorative.

fig, ax = plot_som_architecture(
    som,
    values=medians.reshape(shape),
    cmap="cividis",
    metric_label="Median spectroscopic redshift",
    hatch_mask=(counts < 12).reshape(shape),
    show_neuron_centers=False,
    figsize=(7.2, 4.3),
    depth=0.34,
)
fig.set_label(
    "Four SDSS optical colors feed a hexagonal map. Blue to gold encodes the "
    "median redshift of held-out spectra, and gray hatching indicates fewer than 12 spectra."
)
print(f"{len(train):,} training galaxies and {len(test):,} held-out galaxies.")
print(f"{len(supported)} of {counts.size} cells contain at least 12 held-out spectra.")

# %%
# Compare distributions within two cells
# --------------------------------------
# A cell median hides variation. Use the training spectra to select two
# cells near the 20th and 80th percentiles of median redshift. Require at
# least 12 galaxies in each split for these comparisons. The curves then
# show the full empirical distributions of held-out spectra in those
# cells, with independent plate groups for selection and evaluation.

training_cells = np.ravel_multi_index(som.best_matching_units().T, shape)
training_counts = np.bincount(training_cells, minlength=counts.size)
eligible = supported[training_counts[supported] >= 12]
training_medians = np.array(
    [np.median(redshift[train][training_cells == cell]) for cell in eligible]
)
order = eligible[np.argsort(training_medians, kind="stable")]
selected = order[[len(order) // 5, 4 * len(order) // 5]]
curves = [np.sort(z_test[cells == cell]) for cell in selected]
minimum = max(0.02, np.floor(min(sample[0] for sample in curves) * 100) / 100)
maximum = min(0.35, np.ceil(max(sample[-1] for sample in curves) * 100) / 100)
styles = [
    ("#26669a", "o", "-", "Lower training median"),
    ("#a64d32", "s", "--", "Higher training median"),
]

fig, (ax_map, ax_cdf) = plt.subplots(1, 2, figsize=(8.7, 3.6), layout="constrained")
camera = ObliqueProjection(azimuth=0, elevation=90)
plot_cell_plane(
    counts.reshape(shape),
    ax=ax_map,
    topology=som.topology,
    projection=camera,
    label="Held-out galaxies per cell",
    cmap="Greys",
)
centers = som.lattice.centers().reshape(-1, 2)
screen = camera.project(np.column_stack((centers, np.zeros(len(centers)))))
for cell, sample, (color, marker, linestyle, label) in zip(selected, curves, styles):
    ax_map.scatter(
        *screen[cell],
        s=45,
        marker=marker,
        facecolors="white",
        edgecolors=color,
        lw=1.5,
        zorder=5,
    )
    probability = np.arange(1, len(sample) + 1) / len(sample)
    ax_cdf.step(
        np.r_[minimum, sample, maximum],
        np.r_[0, probability, 1],
        where="post",
        color=color,
        linestyle=linestyle,
        lw=1.7,
        label=f"{label} (n={len(sample)})",
    )
ax_map.set_title("Spectroscopic counts", loc="left", fontsize=11)
ax_cdf.set_title("Two cells in color space", loc="left", fontsize=11)
ax_cdf.set(xlabel="Spectroscopic redshift", ylabel="Cumulative fraction", ylim=(0, 1.03))
ax_cdf.set_xlim(minimum, maximum)
ax_cdf.spines[["top", "right"]].set_visible(False)
ax_cdf.legend(loc="lower right", fontsize=8, frameon=False)
fig.set_label(
    "A count map marks two cells selected using training redshifts. The curves "
    "compare their held-out redshift distributions and show the spread in each cell."
)

# %%
# What the measurements support
# -----------------------------
# These distributions describe bright galaxies with spectra in this
# selection. Overlapping plate footprints can share large-scale structure.
# A redshift distribution for a photometric parent sample also requires
# its cell occupancies and assumptions about which spectra represent it.
# See :doc:`../gallery/plot_photoz_bounds` for an explicitly synthetic
# example of that additional inference problem.
