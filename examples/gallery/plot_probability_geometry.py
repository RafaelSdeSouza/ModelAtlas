"""
Fisher–Rao geometry of photo-z classifiers
==========================================

Fitted classifiers predict whether a galaxy's redshift exceeds the
training-sample median, using four SDSS DR8 optical colors. Each model
is evaluated on identical held-out spectroscopic plates. Fisher–Rao
distances compare Bernoulli predictive distributions; bootstrap connection
frequencies quantify sensitivity to the *held-out reference sample*, not
to re-training, targeting bias, or photo-z calibration error.

A permuted neural network supplies an exact-function-equivalence control.
This example concerns the selected spectroscopic population only.
"""

import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modelatlas.behavior_geometry import build_probability_atlas
from modelatlas.behavior_plot import plot_probability_atlas
from modelatlas.functional import model_predictions

path = (Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent) / "astronomy" / "sdss_dr8_galaxies.npz"
with np.load(path, allow_pickle=False) as cat:
    X, redshift, plates = cat["colors"], cat["redshift"], cat["plate"]
split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
train_idx, ref_idx = next(split.split(X, groups=plates))
rng = np.random.default_rng(1729)
if len(train_idx) > 2700:
    train_idx = np.sort(rng.choice(train_idx, 2700, replace=False))
if len(ref_idx) > 1250:
    ref_idx = np.sort(rng.choice(ref_idx, 1250, replace=False))
threshold = float(np.median(redshift[train_idx]))
scaler = StandardScaler().fit(X[train_idx])
A, B = scaler.transform(X[train_idx]), scaler.transform(X[ref_idx])
y = (redshift[train_idx] > threshold).astype(int)
models = {
    "GLM": LogisticRegression(C=0.9, max_iter=1500).fit(A, y),
    "Regularized GLM": LogisticRegression(C=0.035, max_iter=1500).fit(A, y),
    "MLP A": MLPClassifier(
        hidden_layer_sizes=(12, 7), activation="tanh", solver="lbfgs",
        alpha=0.25, max_iter=650, random_state=2
    ).fit(A, y),
    "Random forest": RandomForestClassifier(
        n_estimators=110, max_depth=5, min_samples_leaf=6,
        random_state=4, n_jobs=1
    ).fit(A, y),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=110, max_depth=5, min_samples_leaf=5,
        random_state=19, n_jobs=1
    ).fit(A, y),
    "RBF SVM": SVC(kernel="rbf", C=2.3, probability=True, random_state=4).fit(A, y),
    "KNN": KNeighborsClassifier(n_neighbors=17).fit(A, y),
    "Gaussian NB": GaussianNB().fit(A, y),
}
permuted = copy.deepcopy(models["MLP A"])
rng = np.random.default_rng(12)
for layer in range(len(permuted.coefs_) - 1):
    order = rng.permutation(permuted.coefs_[layer].shape[1])
    permuted.coefs_[layer] = permuted.coefs_[layer][:, order]
    permuted.intercepts_[layer] = permuted.intercepts_[layer][order]
    permuted.coefs_[layer + 1] = permuted.coefs_[layer + 1][order, :]
models["MLP A (permuted)"] = permuted
predictions = model_predictions(models, B)
families = {
    name: ("MLP" if name.startswith("MLP")
           else "GLM" if "GLM" in name
           else "Forest" if "forest" in name or "Trees" in name
           else "Kernel" if "SVM" in name or "KNN" in name
           else "Bayes")
    for name in predictions
}
atlas = build_probability_atlas(predictions, geometry="fisher_rao", n_bootstrap=250, seed=120)
labels = {
    "GLM": "GLM", "Regularized GLM": "rGLM", "MLP A": "MLP",
    "Random forest": "RF", "Extra Trees": "ET", "RBF SVM": "SVM",
    "KNN": "kNN", "Gaussian NB": "NB"
}
fig, ax = plot_probability_atlas(atlas, family=families, label_map=labels)
fig.set_label(
    "Fisher–Rao distances between redshift-threshold classifiers evaluated on "
    "identical held-out SDSS spectroscopy. Edge strengths encode reference-resampling "
    "stability, not spectroscopic representativeness."
)
out = os.environ.get("MODELATLAS_FIGURE_DIR")
if out:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(target / f"05_probability_geometry.{ext}", dpi=260, bbox_inches="tight")
    edges = [
        {"first": atlas.unique_names[i], "second": atlas.unique_names[j],
         "frequency": float(atlas.edge_frequencies[i, j])}
        for i in range(len(atlas.unique_names))
        for j in range(i + 1, len(atlas.unique_names))
        if atlas.edge_frequencies[i, j] > 0.5
    ]
    (target / "05_probability_geometry.json").write_text(json.dumps({
        "dataset": "SDSS DR8 bright spectroscopic galaxies",
        "redshift_threshold_from_training": threshold,
        "reference_observations": atlas.n_reference,
        "metric": atlas.geometry,
        "n_parameterizations": len(atlas.names),
        "distinct_prediction_vectors": len(atlas.unique_names),
        "bootstrap_replicates": atlas.n_bootstrap,
        "retained_3d": atlas.retained_3d,
        "distance_stress": atlas.distance_stress,
        "stable_edges": edges,
    }, indent=2))
plt.show()
