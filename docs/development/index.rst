Development
===========

The repository follows a ``src/`` layout with Hatchling, PEP 621 metadata,
pytest, and an executable Sphinx gallery. The implementation adapts the
`Scientific Python Library Development Guide <https://learn.scientific-python.org/development/>`__.

.. code-block:: console

   python -m pip install -r requirements/tooling.txt
   pre-commit install
   nox -s lint tests-3.12 docs build

The local suite checks topology, camera geometry, activation symmetries,
attribution conservation, tail bounds against an independent optimization,
and deterministic exports. CI runs Python 3.10 through 3.14 and a separate
3.10 lower-dependency job. Both distribution formats are installed into
separate environments for smoke tests. The gallery build checks rendered
HTML links and the presence of downloadable figure formats.

Add a gallery example
---------------------

Put a standalone ``plot_*.py`` under ``examples/gallery/`` with a
Sphinx-Gallery docstring. Specify the data provenance and the model or
inference assumptions. Set a descriptive label on each Matplotlib figure
with ``fig.set_label(...)`` for the gallery's alternative text. Use fixed
random seeds, an explicit color scale for comparisons, and training-only
preprocessing when evaluating on held-out observations.

Documentation review and hosting
--------------------------------

On a pull request, **Documentation preview** archives the built HTML,
figures and build log in ``modelatlas-docs-preview``. Extract the artifact
and open its ``html/index.html`` (or serve the extracted directory locally).
The planned default project URL is
``https://rafaelsdesouza.github.io/ModelAtlas/``. The repository owner can
choose **GitHub Actions** as the GitHub Pages source after review. The
**Publish approved documentation** workflow requires a manual dispatch
of ``main`` with its ``publish`` input enabled and targets the
``github-pages`` environment. It uses the base URL reported by Pages.

.. toctree::
   :maxdepth: 1

   audit
   documentation
