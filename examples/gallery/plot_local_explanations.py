"""
Baseline-relative explanations for Wine chemistry
=================================================

A two-hidden-layer tanh network predicts class zero versus the two other
Wine classes from eight standardized chemical measurements. Scaling, the
network, and the activation layout use training observations only. The
baseline is the vector of training feature means in standardized coordinates.

Two held-out observations are selected at the extremes of the fitted logit.
Both figures use identical positions and attribution scales. Rust indicates
a positive contribution to the baseline-relative logit and blue a negative
one. Each layer's integrated contributions sum to the logit difference up
to numerical quadrature error.
"""

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from modelatlas import explain_mlp, explanation_layout, plot_explanation_atlas

wine = load_wine()
X_train, X_test, y_train, _ = train_test_split(
    wine.data[:, :8],
    (wine.target == 0).astype(int),
    test_size=0.30,
    random_state=26,
    stratify=wine.target,
)
scaler = StandardScaler().fit(X_train)
train, test = scaler.transform(X_train), scaler.transform(X_test)
model = MLPClassifier(
    hidden_layer_sizes=(16, 9),
    activation="tanh",
    solver="lbfgs",
    alpha=0.35,
    max_iter=1500,
    random_state=31,
).fit(train, y_train)
layout = explanation_layout(model, train)
p = model.predict_proba(test)[:, 1]
explanations = [explain_mlp(model, test[i], steps=512) for i in (np.argmax(p), np.argmin(p))]
scale = max(np.abs(a).max() for item in explanations for a in item.node_contributions[:-1])
edge_scale = max(np.abs(a).max() for item in explanations for a in item.edge_contributions)
features = (
    "Alcohol",
    "Malic acid",
    "Ash",
    "Ash alkalinity",
    "Magnesium",
    "Phenols",
    "Flavanoids",
    "Nonflav. phenols",
)
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
        f"Wine explanation atlas for the observation with the {name} held-out fitted logit. "
        "Eight input features connect to two oblique layers of hidden units, "
        "colored by signed contributions relative to the training-mean baseline."
    )
    print(
        f"{name}: Δlogit = {item.logit_change:+.3f}, "
        f"maximum layer completeness error = {item.max_layer_residual:.2g}"
    )
