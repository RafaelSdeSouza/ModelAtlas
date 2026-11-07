Installation
============

ModelAtlas supports Python 3.10 and later. The distribution is named
``modelatlas-viz`` and the import is ``modelatlas``. Install the development
checkout in an isolated environment.

.. code-block:: console

   git clone https://github.com/RafaelSdeSouza/ModelAtlas.git
   cd ModelAtlas
   git switch dev/research-package-and-docs
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .

On Windows PowerShell, activate with ``.venv\Scripts\Activate.ps1``.

NumPy, SciPy and Matplotlib are runtime dependencies. If you already have a
fitted model, the plotters can inspect the arrays directly. The ``examples``
extra installs scikit-learn and pandas for the original research scripts.
The ``docs`` extra also brings Sphinx, the theme, the gallery and the datasets
used by its figures.

.. code-block:: console

   python -m pip install -e '.[examples]'
   python -c "import modelatlas; print(modelatlas.__version__)"

For the exact documentation and figure environment, see
:doc:`../guide/reproducibility`.
