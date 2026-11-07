"""
A learned prototype graph and a fixed GTM lattice
=================================================

Four Wine measurements are standardized before fitting two unsupervised
representations. The growing neural gas-style graph learns its edges.
Tiles are placed by an orthographic camera applied to three unwhitened
principal components of the prototypes. Hue and width summarize the same
assignment counts. The printed distance stress measures the distortion from
fitted feature space to the rendered plane after one uniform rescaling.

GTM uses a fixed latent lattice and a regularized radial-basis map to feature
space. Its figure colors one prototype component. The mixture responsibilities
are calculated separately.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from modelatlas import (
    fit_prototype_layout,
    growing_neural_gas,
    plot_gtm,
    plot_prototype_graph,
    train_gtm,
)

columns = [0, 1, 6, 9]
labels = ["Alcohol", "Malic acid", "Flavanoids", "Color intensity"]
x = StandardScaler().fit_transform(load_wine().data[:, columns])
graph = growing_neural_gas(x, max_nodes=23, steps=3800, insertion_interval=140, seed=31)
layout = fit_prototype_layout(graph.prototypes)
fig, ax = plot_prototype_graph(graph, graph.assign(x), layout=layout, feature_labels=labels)
fig.set_label(
    "Four Wine features connect to an irregular prototype graph. Tile widths and color "
    "show assignment counts. Fine blue-gray edges connect prototypes learned as neighbors."
)
distortion = layout.diagnostics(graph.prototypes)
print(f"PCA retained variance: {layout.explained_variance_ratio.sum():.3f}")
print(f"Fitted-space to screen distance stress: {distortion.normalized_stress:.3f}")

gtm = train_gtm(x, grid=(6, 7), rbf_grid=(3, 4), n_iter=45)
fig, ax = plot_gtm(gtm, component=2, label="Prototype flavanoids (standardized)")
fig.set_label(
    "Oblique rectangular GTM latent grid colored by the fitted standardized flavanoids component."
)
np.testing.assert_allclose(gtm.responsibilities(x).sum(axis=1), 1)
