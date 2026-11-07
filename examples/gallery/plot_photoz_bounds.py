"""
Spectroscopic selection and identified photo-z ranges
=====================================================

This example uses a **synthetic** target catalogue with noisy photometric
features and selective spectroscopic targeting and success. Only successful
spectroscopic redshifts are supplied to the identification calculation.
The illustrative Wasserstein radius is an assumption measured in redshift
units. Hatched populated cells have no usable spectra. Grey denotes missing
cell statistics.

The population figure varies the assumed radius for the fraction above
redshift 1.2. The shaded region depicts compatible values at each radius.
Its width measures identification ambiguity under the stated assumptions.
"""

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from modelatlas import (
    PhotoZCells,
    fit_som,
    plot_identification_sensitivity,
    plot_photoz_diagnostics,
)

rng = np.random.default_rng(39)
n = 1600
family = rng.binomial(1, 0.40, n)
z = np.clip(rng.normal(0.65 + 0.84 * family, 0.30), 0, None)
photometry = np.column_stack(
    (
        np.log1p(z) + 0.25 * family + rng.normal(0, 0.16, n),
        np.sin(1.7 * z) - 0.30 * family + rng.normal(0, 0.18, n),
        21 + 1.5 * z + rng.normal(0, 0.58, n),
    )
)
attempted = rng.random(n) < expit(1.4 - (photometry[:, 2] - 21))
success = attempted & (rng.random(n) < expit(1.5 - 1.8 * z))
spec = np.where(success, z, np.nan)
x = StandardScaler().fit_transform(photometry)
som = fit_som(x, (7, 8), topology="hexagonal", steps=3600, seed=39)
cells = PhotoZCells(som, x, spec, attempted=attempted)

bounds = cells.identify(z0=1.2, gamma=0.12)
fig, axes = plot_photoz_diagnostics(bounds)
fig.suptitle("Synthetic photo-z catalogue", color="#24354b", fontsize=13)
fig.set_label(
    "Four aligned hexagonal maps of target counts, usable spectra fraction, observed tail fraction "
    "and identified width. The last panel hatches occupied cells without usable spectra."
)
fig, ax = plot_identification_sensitivity(cells, np.linspace(0, 0.45, 13), threshold=1.2)
ax.set_title("Synthetic photo-z catalogue", loc="left", fontsize=11)
fig.set_label(
    "The compatible interval for the fraction above redshift 1.2 widens as the Wasserstein radius increases."
)
print(
    f"Usable spectra: {success.sum()} / {n}; identified interval at radius 0.12: {bounds.population_interval()}"
)
