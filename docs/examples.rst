Examples
========

ModelAtlas is a general-purpose Python visualization package with an
**astronomy-centered gallery** for researchers using machine learning,
self-organizing maps, statistical inference, and explainable AI.

Observational survey applications
---------------------------------

* :doc:`astronomy/plot_sdss_galaxy_som` maps SDSS broadband colors using a
  hexagonal SOM and compares held-out spectroscopic-redshift distributions.
* :doc:`astronomy/plot_sdss_stellar_som` studies stellar temperature and
  iron abundance in the SEGUE color locus.
* :doc:`gallery/plot_local_explanations` interprets an MLP photo-z
  threshold classifier fitted to SDSS galaxy spectroscopy.
* :doc:`gallery/plot_functional_lattice` compares several ML methods for
  the same redshift-classification task, and tracks information from colors.
* :doc:`gallery/plot_functional_regression` compares redshift regressors
  in a common prediction geometry.
* :doc:`gallery/plot_probability_geometry` evaluates model proximity
  through Fisher--Rao distances between their predicted class probabilities.
* :doc:`gallery/plot_prototype_graph` compares adaptive graph and GTM
  representations of SEGUE stellar colors.
* :doc:`gallery/plot_aligned_networks` compares independent neural
  networks fitted to historical solar activity.

These are reproducible educational studies, **not** scientific validation
by SDSS, Euclid, Rubin, or other survey collaborations. Catalogue cuts,
splits and provenance are documented in :doc:`guide/sdss_data`.

Time-domain astronomy and population inference
----------------------------------------------

Historical solar activity supports a real time-ordered network-alignment
example in :doc:`gallery/plot_aligned_networks`. The
:doc:`gallery/plot_photoz_bounds` tutorial uses **synthetic galaxies** to
illustrate how spectroscopic selection and sensitivity assumptions affect
an identified interval. It must not be confused with survey-validated
population-redshift calibration.

See :doc:`guide/communities` for research workflows relevant to
Rubin/LSST, Euclid, stellar-population analyses, and alert brokers.

.. toctree::
   :hidden:

   Observational astronomy <astronomy/index>
   Model and method gallery <gallery/index>
