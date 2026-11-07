"""
Symmetry-aware neural networks on observed solar activity
=========================================================

Two independently fitted tanh networks predict whether the next annual
sunspot count exceeds the most recent count. The inputs are eight successive
annual values from the historical NOAA record (1700--2008), distributed by
statsmodels. A chronological split avoids random mixing of future labels
into the training set; adjacent windows still share historical measurements.

Hidden units are matched by signed activation traces *on training windows*.
This aligns parameterizations without modifying the fitted predictions.
The figures show where two independently trained models disagree; matching
neurons does not establish predictive equivalence.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.datasets import sunspots

from modelatlas import (
    match_tanh_mlp,
    plot_alignment_diagnostic,
    plot_independent_pair,
    plot_residual_atlas,
)

activity = np.asarray(sunspots.load().data["SUNACTIVITY"], dtype=float)
windows = np.lib.stride_tricks.sliding_window_view(activity, 9)
X = windows[:, :8]
y = (windows[:, 8] > windows[:, 7]).astype(int)
cut = int(0.72 * len(X))
scaler = StandardScaler().fit(X[:cut])
train, test = scaler.transform(X[:cut]), scaler.transform(X[cut:])
y_train, y_test = y[:cut], y[cut:]
first, second = [
    MLPClassifier(
        hidden_layer_sizes=(16, 9),
        activation="tanh",
        solver="lbfgs",
        alpha=0.75,
        max_iter=1600,
        random_state=seed,
    ).fit(train, y_train)
    for seed in (4, 29)
]
aligned = match_tanh_mlp(first, second, train, X_check=test)
p, q = (model.predict_proba(test)[:, 1] for model in (first, second))
obs = int(np.argmax(np.abs(p - q)))

fig, axes = plot_independent_pair(
    first,
    aligned,
    test[obs],
    title="Annual solar activity · two independently fitted networks",
    label="Next year rises",
    metrics={"test_auc_A": roc_auc_score(y_test, p), "test_auc_B": roc_auc_score(y_test, q)},
)
fig.set_label(
    "Two networks trained independently on historical solar activity. "
    "Hidden units are matched by training responses; the held-out predictions "
    "can still differ for the same eight-year input window."
)

fig, ax = plot_residual_atlas(first, aligned, test, "Solar activity · held-out activation residuals")
fig.set_label(
    "Residual neuron activation differences on the chronologically held-out solar observations. "
    "Neuron arrangement is aligned, not an intrinsic physical lattice."
)

fig, axes = plot_alignment_diagnostic(first, second, aligned, train, test, y_test)
fig.set_label(
    "Solar activity: training activation correspondence before and after "
    "signed neuron alignment, compared with prediction disagreement on later years."
)
print(f"Maximum alignment-induced change in second-model predictions: "
      f"{aligned.prediction_preservation_error:.2g}")
