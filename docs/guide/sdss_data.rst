SDSS catalogues and sample selection
======================================

The two :doc:`../astronomy/index` examples use public SDSS observations.
The source tables come from astroML and the documentation builds from
the two bundled subsets. See the SDSS `data policy and credits
<https://www.sdss.org/collaboration/image-use-policy/>`__ when reusing them.

Galaxies · SDSS DR8
---------------------

The input is the `astroML spectroscopic galaxy table
<https://www.astroml.org/modules/generated/astroML.datasets.fetch_sdss_specgals.html>`__
from SDSS DR8, described by
`Aihara et al. (2011, ApJS 193, 29) <https://arxiv.org/abs/1101.1559>`__.
The original query matches spectroscopic objects to imaging and joins the
galaxy spectroscopy tables.

We retain Petrosian :math:`14 \le r \le 17.77` mag, foreground
:math:`0 \le A_r \le 0.15` mag, and pipeline redshifts
:math:`0.02 \le z \le 0.35` with :math:`0 < \sigma_z \le 0.001`.
Model-magnitude errors must be positive and at most 0.15 mag in
:math:`u` and 0.08 mag in :math:`griz`. For repeated imaging IDs we retain
the smallest redshift error, with source row breaking ties. The cuts
leave 500,188 unique objects. We take 12,000 without replacement using
NumPy's PCG64 generator with seed 20261107.

The SOM inputs are differences of the observed model magnitudes
:math:`u-g`, :math:`g-r`, :math:`r-i`, and :math:`i-z` in magnitudes.
The low-extinction cut limits Galactic reddening. Spectroscopic redshift
is the quantity summarized after the map is fit. In the case study we
hold out 25% of plate IDs using split seed 18.

Stars · SDSS DR9 SEGUE
------------------------

The input is the `astroML SEGUE Stellar Parameter Pipeline table
<https://www.astroml.org/modules/generated/astroML.datasets.fetch_sdss_sspp.html>`__
from SDSS DR9, SSPP rerun 122. The survey release is described by
`Ahn et al. (2012, ApJS 203, 21) <https://arxiv.org/abs/1207.7137>`__.

From the observed PSF magnitudes, subtract
:math:`(A_u,A_g,A_r)=(1.810,1.400,1)A_r`, following the coefficients in
astroML's SSPP example based on
`Berry et al. (2012, ApJ 757, 166) <https://arxiv.org/abs/1111.4985>`__.
The SOM uses the two corrected colors :math:`(u-g)_0` and :math:`(g-r)_0`.
We retain :math:`0 < (u-g)_0 < 3`, :math:`0.20 \le (g-r)_0 \le 0.60`,
:math:`15 \le g_0 < 19.5` mag, :math:`0 \le A_r \le 0.20` mag,
and :math:`\log g \ge 4` (cgs). Effective temperatures are between
5,000 and 7,000 K and :math:`[\mathrm{Fe/H}]` is between −3 and 0.5 dex.
Metallicity errors are positive and at most 0.20 dex, spectral
signal-to-noise is at least 20, and photometric errors are positive and
at most 0.10 mag in :math:`u` and 0.05 mag in :math:`g,r`.

Within 1 arcsec, discard a record if a neighbor has higher spectral
signal-to-noise. Source row breaks ties. This leaves 74,912 records,
from which we take 8,000 without replacement with seed 20261107. The
case study holds out 25% of plate IDs with split seed 29. The plate
holdout separates fitting and evaluation. The pipeline parameters
and catalogue selection define the reference sample for the comparison.

Regenerate the bundled subsets
--------------------------------

The data preparation script verifies SHA-256 for all three upstream files.
The :doc:`gallery build <reproducibility>` reads the resulting archives
locally. Astropy is needed only to recreate them from FITS. Run these commands
from the repository root in a Python 3.12 environment.

.. code-block:: console

   python -m pip install 'astropy==7.1.1' 'numpy==2.3.5' 'scipy==1.17.0'
   mkdir -p /tmp/modelatlas-sdss
   source_url=https://raw.githubusercontent.com/astroML/astroML-data/869ca28294159d4c6302c1e63b9402bd54abf29f/datasets
   for file in SDSSspecgalsDR8_1.fit.gz SDSSspecgalsDR8_2.fit.gz SDSSssppDR9_rerun122.fit.gz; do
       curl --fail --location "$source_url/$file" --output "/tmp/modelatlas-sdss/$file"
   done
   python scripts/build_sdss_gallery_data.py /tmp/modelatlas-sdss

Each archive stores colors, plate ID, the spectroscopic quantities used in
the figures, and the zero-based source row. Galaxy rows index the first
FITS file followed by the second. The stellar rows index the SSPP FITS file.
The example pages link to the exact archives required by their scripts and
notebooks.

* ``sdss_dr8_galaxies.npz`` SHA-256
  ``5f113dbbe8cf02770ed88a1efafd35e9810f228648c9403f72e664cd3f6bd939``
* ``sdss_dr9_stars.npz`` SHA-256
  ``56ce781967295a77030b2d72f58d95b5f467664ab94076d0f5efae5b5e3d9bb0``
