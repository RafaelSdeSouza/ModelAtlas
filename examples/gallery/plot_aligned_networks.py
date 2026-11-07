"""
Aligning independent neural networks on real measurements
=========================================================

Two binary tanh networks are fitted with different seeds to a common
stratified training sample from the Wisconsin Diagnostic Breast Cancer
dataset. Hidden units are matched by signed activation profiles on
training inputs. This reparameterization preserves the second fitted
function. The pair shows the held-out observation with the largest
probability difference, and the residual atlas uses all held-out inputs.
This single-split example illustrates model comparison.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from modelatlas import (
    match_tanh_mlp,
    plot_alignment_diagnostic,
    plot_independent_pair,
    plot_residual_atlas,
)

data = load_breast_cancer()
columns = [0, 1, 2, 3, 6, 7, 20, 27]
X_train, X_test, y_train, y_test = train_test_split(
    data.data[:, columns],
    (data.target == 0).astype(int),
    test_size=0.31,
    random_state=2026,
    stratify=data.target,
)
scaler = StandardScaler().fit(X_train)
train, test = scaler.transform(X_train), scaler.transform(X_test)
first, second = [
    MLPClassifier(
        hidden_layer_sizes=(16, 9),
        activation="tanh",
        solver="lbfgs",
        alpha=0.6,
        max_iter=1500,
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
    title="Wisconsin measurements · two fitted classifiers",
    label="malignant",
    metrics={"test_auc_A": roc_auc_score(y_test, p), "test_auc_B": roc_auc_score(y_test, q)},
)
fig.set_label(
    "Two tanh networks with identical neuron slots. Blue and rust show signed activations. "
    "The annotated probabilities differ even after matching."
)
fig, ax = plot_residual_atlas(first, aligned, test, "Wisconsin measurements · held-out residuals")
fig.set_label(
    "Oblique residual network with brighter tiles for larger held-out RMS activation differences."
)
fig, axes = plot_alignment_diagnostic(first, second, aligned, train, test, y_test)
fig.set_label(
    "Training-sample activation correlations before and after alignment, beside held-out probability agreement."
)
print(f"Candidate probability preservation error: {aligned.prediction_preservation_error:.2g}")
