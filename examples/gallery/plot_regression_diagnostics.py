"""
Regression uncertainty and GLM coefficients
===========================================

For a simulated straight-line calibration with independent, constant-variance
Gaussian errors, the Student-t prediction interval is pointwise for a new
response. It includes residual scatter and coefficient estimation error.
The example computes the interval before passing it to the plotting API.

The second figure shows fitted slopes from binary logistic regression on
standardized Wine chemistry. Its link is the logit. One unit on the feature
scale is one sample standard deviation used by the fitted scaler.
"""

import numpy as np
from scipy.stats import t
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from modelatlas import plot_coefficients, plot_predictions

rng = np.random.default_rng(281)
x = rng.uniform(0, 3, 65)
y = 1.3 + 0.8 * x + rng.normal(0, 0.38, len(x))
design = np.column_stack((np.ones(len(x)), x))
coef = np.linalg.lstsq(design, y, rcond=None)[0]
sigma = np.sqrt(np.sum((y - design @ coef) ** 2) / (len(y) - 2))
grid = np.linspace(0, 3, 24)
new_design = np.column_stack((np.ones(len(grid)), grid))
prediction = new_design @ coef
observed = 1.3 + 0.8 * grid + rng.normal(0, 0.38, len(grid))
leverage = np.sum((new_design @ np.linalg.inv(design.T @ design)) * new_design, axis=1)
margin = t.ppf(0.975, len(y) - 2) * sigma * np.sqrt(1 + leverage)
fig, ax = plot_predictions(
    observed,
    prediction,
    x=grid,
    xlabel="Simulated calibration input",
    ylabel="Response",
    intervals=(prediction - margin, prediction + margin),
    interval_label="95% pointwise prediction interval",
)
fig.set_label(
    "Observed and fitted responses in a simulated calibration, with a labeled pointwise prediction band."
)

wine = load_wine()
columns = [0, 1, 6, 9]
features = StandardScaler().fit_transform(wine.data[:, columns])
glm = LogisticRegression(C=1, random_state=24).fit(features, wine.target == 0)
fig, ax = plot_coefficients(
    glm,
    labels=("Alcohol", "Malic acid", "Flavanoids", "Color intensity"),
    xlabel="Change in fitted log-odds per standard deviation",
)
fig.set_label(
    "Signed slopes for four standardized Wine measurements, displayed on the fitted log-odds scale."
)
