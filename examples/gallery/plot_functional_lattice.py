"""
Comparing photo-z classifiers and information in SDSS colors
============================================================

Fit several ML model families to predict whether a galaxy's redshift
exceeds the training-plate median from the four optical SDSS DR8 colors.
Spectroscopic plates are split before fitting any preprocessing or models.
A common set of held-out galaxies determines the geometry between models.

The empirical information lattice studies how much variation in a *fitted
model's predictions* remains after revealing combinations of four binned
colors. It does not measure redshift uncertainty, inferential identification,
or the representativeness of the bright spectroscopic reference sample.
A permuted MLP demonstrates an exact symmetry without making the other
independently trained models equivalent.
"""

import copy
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modelatlas.functional import (
    find_empirical_equivalences,
    functional_map,
    information_lattice,
    model_predictions,
    monotonicity_violations,
)
from modelatlas.functional_plot import (
    plot_functional_atlas,
    plot_information_lattice,
    plot_prediction_fibers,
)


def build():
    path = Path(__file__).resolve().parent.parent / "astronomy" / "sdss_dr8_galaxies.npz"
    with np.load(path, allow_pickle=False) as catalogue:
        X = catalogue["colors"]
        z = catalogue["redshift"]
        plates = catalogue["plate"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
    train_idx, test_idx = next(split.split(X, groups=plates))
    rng = np.random.default_rng(1729)
    if len(train_idx) > 2700:
        train_idx = np.sort(rng.choice(train_idx, 2700, replace=False))
    if len(test_idx) > 1250:
        test_idx = np.sort(rng.choice(test_idx, 1250, replace=False))
    threshold = float(np.median(z[train_idx]))
    y_train = (z[train_idx] > threshold).astype(int)
    y_ref = (z[test_idx] > threshold).astype(int)
    scaler = StandardScaler().fit(X[train_idx])
    A, B = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    models = {
        "GLM": LogisticRegression(C=0.9, max_iter=1500).fit(A, y_train),
        "Regularized GLM": LogisticRegression(C=0.035, max_iter=1500).fit(A, y_train),
        "MLP A": MLPClassifier(
            hidden_layer_sizes=(12, 7), activation="tanh", solver="lbfgs",
            alpha=0.25, max_iter=650, random_state=2
        ).fit(A, y_train),
        "Random forest": RandomForestClassifier(
            n_estimators=110, max_depth=5, min_samples_leaf=6,
            random_state=4, n_jobs=1
        ).fit(A, y_train),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=110, max_depth=5, min_samples_leaf=5,
            random_state=19, n_jobs=1
        ).fit(A, y_train),
        "RBF SVM": SVC(
            kernel="rbf", C=2.3, probability=True, random_state=4
        ).fit(A, y_train),
        "KNN": KNeighborsClassifier(n_neighbors=17).fit(A, y_train),
        "Gaussian NB": GaussianNB().fit(A, y_train),
    }
    clone = copy.deepcopy(models["MLP A"])
    rng = np.random.default_rng(12)
    for layer in range(len(clone.coefs_) - 1):
        permutation = rng.permutation(clone.coefs_[layer].shape[1])
        clone.coefs_[layer] = clone.coefs_[layer][:, permutation]
        clone.intercepts_[layer] = clone.intercepts_[layer][permutation]
        clone.coefs_[layer + 1] = clone.coefs_[layer + 1][permutation, :]
    models["MLP A (permuted)"] = clone
    predicted = model_predictions(models, B)
    atlas = functional_map(predicted)
    families = {
        name: ("MLP" if name.startswith("MLP")
               else "GLM" if "GLM" in name
               else "Forest" if "forest" in name or "Trees" in name
               else "Kernel" if "SVM" in name or "KNN" in name
               else "Bayes")
        for name in predicted
    }
    brier = {name: brier_score_loss(y_ref, p) for name, p in predicted.items()}
    auc = {name: roc_auc_score(y_ref, p) for name, p in predicted.items()}
    cuts = np.median(A, axis=0)
    # U/G/R/I abbreviate u-g, g-r, r-i and i-z; they are not passband labels.
    lattice = information_lattice(
        B, predicted["MLP A"], feature_indices=(0, 1, 2, 3),
        thresholds=cuts, feature_names=("U", "G", "R", "I")
    )
    return B, y_ref, predicted, atlas, lattice, families, brier, auc, threshold


B, y, predictions, atlas, lattice, families, brier, auc, threshold = build()
destination = os.environ.get("MODELATLAS_FIGURE_DIR")
out = Path(destination) if destination else None
if out is not None:
    out.mkdir(parents=True, exist_ok=True)

fig, ax = plot_functional_atlas(atlas, family=families, quality=brier)
fig.set_label(
    "Nine fitted parameterizations of photo-z threshold classifiers on held-out SDSS "
    "galaxy colors. Distance reflects agreement on the same reference spectra; "
    "the two hidden-neuron permutations coincide."
)
if out is not None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"01_functional_atlas.{ext}", dpi=260, bbox_inches="tight")
plt.show()

fig, ax = plot_information_lattice(lattice)
fig.set_label(
    "Information lattice for four SDSS color indices: U denotes u-g, G denotes "
    "g-r, R denotes r-i, and I denotes i-z. Prism height shows the unresolved "
    "variation of one neural redshift classifier's predictions."
)
if out is not None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"02_information_lattice.{ext}", dpi=260, bbox_inches="tight")
plt.show()

fig, ax = plot_prediction_fibers(
    B, predictions["MLP A"], features=(0, 1),
    labels=(r"$u-g$", r"$g-r$")
)
fig.set_label(
    "Within color-bin variation of predicted high-redshift probabilities from a "
    "fixed MLP when the other SDSS colors are not displayed. The intervals are "
    "not confidence or photo-z identification bounds."
)
if out is not None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"03_projection_fibers.{ext}", dpi=260, bbox_inches="tight")
plt.show()

if out is not None:
    a, b = atlas.names.index("MLP A"), atlas.names.index("MLP A (permuted)")
    stats = {
        "reference_sample": "held-out SDSS DR8 bright spectroscopic galaxies",
        "reference_size": len(B),
        "redshift_threshold_from_training": threshold,
        "model_parameterizations": len(predictions),
        "distance_stress_3d": atlas.stress3,
        "retained_variation_3d": atlas.variance_retained_3d,
        "permutation_prediction_max_difference":
            float(np.max(np.abs(predictions["MLP A"] - predictions["MLP A (permuted)"]))),
        "permutation_distance": float(atlas.distances[a, b]),
        "empirical_equivalences": [
            [atlas.names[i] for i in group]
            for group in find_empirical_equivalences(atlas)
        ],
        "information_monotonicity_violations": len(monotonicity_violations(lattice)),
        "baseline_prediction_variance": lattice.baseline_variance,
        "prediction_variance_after_four_binned_colors":
            lattice.nodes[-1].residual_variance,
        "shapley_of_variance_reduction": dict(zip(
            lattice.feature_names, map(float, lattice.shapley_variance_reduction)
        )),
        "held_out_brier": brier,
        "held_out_auc": auc,
    }
    (out / "diagnostics.json").write_text(json.dumps(stats, indent=2))
print(f"Reference galaxies: {len(B)}; training-derived threshold: {threshold:.3f}")
