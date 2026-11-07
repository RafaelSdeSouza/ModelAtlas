"""
Regression models of galaxy spectroscopic redshift
=================================================

Ridge, forests, kernel regression and k-nearest neighbors predict SDSS
spectroscopic redshift from the four observed optical colors. The reference
galaxies occupy held-out spectroscopic plates; all prediction vectors use
the *same* reference galaxy ordering and the same redshift units.

ModelAtlas places entire fitted regressors by functional agreement, not by
the structure or number of their internal parameters. This selected bright
spectroscopic sample cannot establish calibration for a photometric parent.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from modelatlas.functional import functional_map, model_predictions
from modelatlas.functional_plot import plot_functional_atlas

path = Path(__file__).resolve().parent.parent / "astronomy" / "sdss_dr8_galaxies.npz"
with np.load(path, allow_pickle=False) as cat:
    X, z, plates = cat["colors"], cat["redshift"], cat["plate"]
split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
train_idx, test_idx = next(split.split(X, groups=plates))
rng = np.random.default_rng(127)
if len(train_idx) > 2800:
    train_idx = np.sort(rng.choice(train_idx, 2800, replace=False))
if len(test_idx) > 1250:
    test_idx = np.sort(rng.choice(test_idx, 1250, replace=False))
A, B = X[train_idx], X[test_idx]
ya, yb = z[train_idx], z[test_idx]
models = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=12.0)).fit(A, ya),
    "Random forest": RandomForestRegressor(
        n_estimators=110, min_samples_leaf=6, random_state=1, n_jobs=1
    ).fit(A, ya),
    "SVR": make_pipeline(StandardScaler(), SVR(C=0.12, epsilon=0.01)).fit(A, ya),
    "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(
        n_neighbors=21
    )).fit(A, ya),
    "Boosting": HistGradientBoostingRegressor(
        max_iter=110, random_state=1, max_leaf_nodes=6
    ).fit(A, ya),
}
predictions = model_predictions(models, B)
atlas = functional_map(predictions)
families = {
    "Ridge": "GLM", "Random forest": "Forest", "SVR": "Kernel",
    "kNN": "Kernel", "Boosting": "Forest"
}
fig, ax = plot_functional_atlas(atlas, family=families, figsize=(7.5, 5.8))
fig.set_label(
    "Functional geometry of five fitted redshift regressors on the same held-out "
    "SDSS galaxies. The figure shows model prediction similarity, not a redshift "
    "error or uncertainty interval."
)
out = os.environ.get("MODELATLAS_FIGURE_DIR")
if out:
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(target / f"04_regression_functional_atlas.{ext}", dpi=260, bbox_inches="tight")
plt.show()
print(f"Held-out galaxy regressors: {len(B)}; 3D stress: {atlas.stress3:.3f}")
