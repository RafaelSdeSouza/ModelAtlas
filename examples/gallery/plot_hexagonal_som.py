"""
Square and hexagonal maps of historical solar activity
======================================================

The input comprises 301 overlapping windows from the annual NOAA sunspot
record (1700–2008), bundled with statsmodels. Each window has five entries
at two-year intervals and is scaled with one common mean and standard
deviation. The map is a description of historical windows. A centered
window includes future values relative to its central year.

Both fits use the same observations, grid size, color range, and random
seed. The topology changes the training neighborhood itself. Color gives
the fitted prototype's central annual sunspot count in original units.
Each strand represents one input component of one prototype. Prism
thickness is decorative. The circle highlights the best-matching prototype
for the window with the highest central count.
"""

import numpy as np
from statsmodels.datasets import sunspots

from modelatlas import SOMMap, fit_som, plot_metric, plot_som_architecture

counts = np.asarray(sunspots.load().data["SUNACTIVITY"], dtype=float)
profiles = np.lib.stride_tricks.sliding_window_view(counts, 9)[:, ::2]
center, spread = profiles.mean(), profiles.std()
scaled = (profiles - center) / spread
labels = ("t − 4", "t − 2", "t", "t + 2", "t + 4")
selected = int(np.argmax(profiles[:, 2]))

fits = [
    fit_som(scaled, (8, 8), steps=4200, seed=21, topology=topology)
    for topology in ("hexagonal", "rectangular")
]
prototype_counts = np.stack([f.codebook[..., 2] * spread + center for f in fits])
color_range = (
    25 * np.floor(prototype_counts.min() / 25),
    25 * np.ceil(prototype_counts.max() / 25),
)
for fitted in fits:
    topology = fitted.topology
    som = SOMMap(
        fitted.codebook,
        scaled,
        labels,
        fitted.codebook * spread + center,
        topology=fitted.topology,
    )
    cell = tuple(som.best_matching_units(scaled[[selected]])[0])
    fig, ax = plot_som_architecture(
        som,
        values=som.display_codebook[..., 2],
        metric_label="Central annual sunspot count",
        highlight=cell,
        cmap="cividis",
        value_range=color_range,
        figsize=(7.2, 4.5),
        show_neuron_centers=False,
    )
    fig.set_label(
        f"{topology.capitalize()} prism lattice with five linked input features. "
        "Blue to amber shows the central annual sunspot count. "
        "A white circle identifies the prototype for the largest observed central count."
    )
    diagnostics = som.diagnostics()
    print(
        f"{topology}: quantization error = {diagnostics['quantization_error']:.3f}, "
        f"topographic error = {diagnostics['topographic_error']:.3f}"
    )

fig, ax = plot_metric(som, "u_matrix", cmap="magma")
fig.set_label(
    "Top-down square U-matrix. Brighter cells have larger fitted-space distances to their four neighbors."
)
