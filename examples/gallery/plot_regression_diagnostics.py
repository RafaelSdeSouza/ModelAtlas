"""
Inspecting statistical models of galaxy redshift
================================================

Ridge regression predicts spectroscopic redshift from four SDSS galaxy
colors; held-out plate-group galaxies supply the predictive check.
The second figure is a logistic GLM fitted to the same selected
spectroscopic population. No calibration uncertainty is claimed.
"""

from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from modelatlas import plot_coefficients, plot_predictions

path = Path(__file__).resolve().parent.parent / "astronomy" / "sdss_dr8_galaxies.npz"
with np.load(path, allow_pickle=False) as catalogue:
    X = catalogue["colors"]
    z = catalogue["redshift"]
    plates = catalogue["plate"]
split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=18)
train_idx, test_idx = next(split.split(X, groups=plates))
scaler = StandardScaler().fit(X[train_idx])
A, B = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
ridge = Ridge(alpha=12.0).fit(A, z[train_idx])
pred = ridge.predict(B)
ordered = np.argsort(z[test_idx], kind="stable")
indices = ordered[np.linspace(0, len(ordered) - 1, 120, dtype=int)]
fig, ax = plot_predictions(
    z[test_idx][indices], pred[indices],
    x=np.arange(len(indices)),
    xlabel="Held-out galaxies sorted by spectroscopic redshift",
    ylabel="Redshift",
)
fig.set_label(
    "Ridge-predicted versus observed spectroscopic redshift for held-out "
    "SDSS DR8 galaxies. No uncertainty interval is asserted."
)
threshold = float(np.median(z[train_idx]))
target = (z[train_idx] > threshold).astype(int)
glm = LogisticRegression(C=1.0, max_iter=1500, random_state=24).fit(A, target)
fig, ax = plot_coefficients(
    glm, labels=("u-g", "g-r", "r-i", "i-z"),
    xlabel="Change in fitted log-odds per standardized color",
)
fig.set_label(
    "Signed logistic photo-z threshold coefficients trained on selected "
    "SDSS DR8 spectroscopy; the coefficients are not causal effects."
)
