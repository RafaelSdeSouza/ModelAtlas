Documentation references
========================

Five scientific Python documentation sites inform the structure of ModelAtlas.

.. list-table::
   :header-rows: 1
   :widths: 19 40 41

   * - Reference
     - Pattern to borrow
     - How ModelAtlas uses it
   * - `Astropy <https://docs.astropy.org/en/stable/>`__
     - Separate getting started, the user guide, and longer learning material.
     - Give installation, scientific interpretation, examples, and the API
       distinct entry points.
   * - `galpy <https://docs.galpy.org/en/stable/>`__
     - Lead with an astronomical task and show a result early.
     - Open with the SDSS color–redshift map and give each observational
       example a scientific question.
   * - `astroML <https://www.astroml.org/examples/datasets/index.html>`__
     - Describe the survey data alongside runnable examples.
     - Document catalogue provenance, units, cuts, and the train/test split
       beside the plot.
   * - `scikit-learn <https://scikit-learn.org/stable/auto_examples/>`__
     - Group a large gallery by purpose and distinguish method demos from
       application tutorials.
     - Give observational studies and model diagnostics separate sections.
   * - `Matplotlib <https://matplotlib.org/stable/gallery/>`__
     - Connect a visible example directly to its figure and source.
     - Pair each figure with its SVG, PDF, and PNG exports. Place
       interpretation and data downloads on the same page.

When adding a case study, state the scientific question, link the catalogue
and its citation, explain which observations fit the model, and say what the
visual encodings measure. Check the gallery and example page at phone,
tablet, and desktop widths. Keep figure previews and their captions in
separate layout rows.
