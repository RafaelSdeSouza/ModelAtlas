Examples
========

Choose a scientific case study to work from observations, or browse the
model gallery for a specific plotting function.

Observational astronomy
-----------------------

* :doc:`astronomy/plot_sdss_galaxy_som` maps four SDSS colors onto a hexagonal
  lattice and compares the distributions of held-out spectroscopic redshifts.
* :doc:`astronomy/plot_sdss_stellar_som` examines how two stellar colors vary
  with effective temperature and metallicity in SDSS/SEGUE.

Each study comes with the catalogue subset, the analysis code, and vector
figures. :doc:`astronomy/index` collects the observational examples.

Galaxy-color SOMs also support survey redshift calibration, as in
`Buchs et al. (2019) <https://arxiv.org/abs/1901.05005>`__. The SDSS
case study illustrates the color–redshift mapping step using spectra
on held-out plates.

Models and diagnostics
----------------------

The :doc:`gallery/index` covers prototype graphs, neural explanations and
alignment, regression, solar activity, and a synthetic exercise on how
spectroscopic selection affects partial identification.

.. toctree::
   :hidden:

   Astronomy case studies <astronomy/index>
   Model and method gallery <gallery/index>
