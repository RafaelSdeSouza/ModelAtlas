"""
Explaining a neural photo-z classifier on SDSS galaxy colors
============================================================

A small tanh neural network predicts whether a galaxy's spectroscopic redshift
is above the *training-sample median* given the observed SDSS DR8 colors
u-g, g-r, r-i, i-z. Entire spectroscopic plates are kept on one side of the
train/test split. This is an illustration of what a fitted model learned
about the selected bright spectroscopic population, not calibration of the
full photometric parent sample.

Integrated path attributions trace how a selected observation changes the
fitted log-odds relative to the training-mean input. Red and blue indicate
signed contributions, not astrophysical causal effects. The two figures
share neuron coordinates and attribution scales for direct comparison.
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from modelatlas import explain_mlp, explanation_layout, plot_explanation_atlas

directory = (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent) / "astronomy"
with np.load(directory / "sdss_dr8_galaxies.npz", allow_pickle=False) as data:
    colors = data["colors"]
    redshift = data["redshift"]
    plates = data["plate"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
idx_train, idx_test = next(splitter.split(colors, groups=plates))
threshold = float(np.median(redshift[idx_train]))
y_train = (redshift[idx_train] > threshold).astype(int)
scaler = StandardScaler().fit(colors[idx_train])
train, test = scaler.transform(colors[idx_train]), scaler.transform(colors[idx_test])

model = MLPClassifier(
    hidden_layer_sizes=(16, 9),
    activation="tanh",
    solver="lbfgs",
    alpha=0.5,
    max_iter=1500,
    random_state=31,
).fit(train, y_train)
layout = explanation_layout(model, train)
p = model.predict_proba(test)[:, 1]
selected = (int(np.argmax(p)), int(np.argmin(p)))
explanations = [explain_mlp(model, test[i], steps=512) for i in selected]
scale = max(np.abs(a).max() for item in explanations for a in item.node_contributions[:-1])
edge_scale = max(np.abs(a).max() for item in explanations for a in item.edge_contributions)
features = (r"$u-g$", r"$g-r$", r"$r-i$", r"$i-z$")

for name, item in zip(("highest", "lowest"), explanations):
    fig, ax = plot_explanation_atlas(
        model,
        item,
        grid_layout=layout,
        feature_names=features,
        attribution_scale=scale,
        edge_scale=edge_scale,
    )
    fig.set_label(
        f"SDSS DR8 color-based neural redshift classifier: {name} held-out predicted "
        f"probability of z exceeding the training median ({threshold:.3f}). "
        "Signed path attributions refer to the training-mean baseline. "
        "The spectroscopic sample does not establish population representativeness."
    )
    print(
        f"{name}: Δlogit = {item.logit_change:+.3f}, "
        f"maximum layer completeness error = {item.max_layer_residual:.2g}"
    )
