"""
Adaptive prototype graphs in observed stellar-color space
=========================================================

A Growing Neural Gas-style graph learns connections between regions in the
two-dimensional SEGUE stellar-color locus. GTM provides an alternative with
a fixed latent grid. Both representations use photometry alone; stellar
temperature and metallicity are not used to learn their prototypes.

The sample is the selected SDSS/SEGUE spectroscopic reference, not a
representative Milky Way stellar census.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from modelatlas import (fit_prototype_layout, growing_neural_gas,
                        plot_gtm, plot_prototype_graph, train_gtm)

path = (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent) / "astronomy" / "sdss_dr9_stars.npz"
with np.load(path, allow_pickle=False) as cat:
    colors, plates = cat["colors"], cat["plate"]
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=29)
train_idx, test_idx = next(splitter.split(colors, groups=plates))
scaler = StandardScaler().fit(colors[train_idx])
X = scaler.transform(colors[train_idx])
X_test = scaler.transform(colors[test_idx])
rng = np.random.default_rng(123)
if len(X) > 3300:
    X = X[rng.choice(len(X), 3300, replace=False)]
if len(X_test) > 900:
    X_test = X_test[rng.choice(len(X_test), 900, replace=False)]
graph = growing_neural_gas(X, max_nodes=23, steps=3800, insertion_interval=140, seed=31)
layout = fit_prototype_layout(graph.prototypes)
fig, ax = plot_prototype_graph(
    graph, graph.assign(X_test), layout=layout,
    feature_labels=["(u-g)0", "(g-r)0"]
)
fig.set_label(
    "Adaptive learned graph of observed SEGUE stellar colors. Prism color and "
    "size encode held-out stellar occupancy. Edges represent learned adjacency."
)
distortion = layout.diagnostics(graph.prototypes)
print(f"Prototype-layout projection stress: {distortion.normalized_stress:.3f}")
gtm = train_gtm(X, grid=(6, 7), rbf_grid=(3, 4), n_iter=45)
fig, ax = plot_gtm(gtm, component=0, label="Prototype (u-g)0 (standardized)")
fig.set_label(
    "GTM of observed SEGUE stellar colors. The latent grid is colored "
    "by its learned first-color component, not a measured stellar parameter."
)
np.testing.assert_allclose(gtm.responsibilities(X_test).sum(axis=1), 1)
