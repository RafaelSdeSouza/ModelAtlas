# ModelAtlas

**Research preview — experimental scientific visualization software.** ModelAtlas renders fitted-model structure and diagnostics as reproducible Matplotlib figures (SVG, PDF, PNG). It currently includes SOM codebook lattices and connections, regression coefficients, small feed-forward networks, and synthetic photo-z missing-data diagnostics.

The project is under active methodological development. The visual-grammar experiments in `examples/visual_grammar/` illustrate how high-dimensional models can be inspected through projection loss, local gradients, data proximity, assumption-dependent identified bounds, and disagreement between fitted models.

**Important:** The photo-z example uses synthetic data. The plug-in bounds and visual encodings are demonstrations, not validated survey inference, statistical confidence intervals, or claims of novel inferential theory.

Install locally with `pip install -e '.[examples,test]'` and run `pytest -q`. See `SCIENTIFIC_NOTES.md`, `REPRODUCIBILITY.md`, and the examples for assumptions and limitations.

Repository development is **public**. Do not upload unpublished observations or paper-specific results without an explicit decision to release them. A citeable versioned software release is planned after methodological review.

Maintainer: Rafael S. de Souza. License: MIT.
