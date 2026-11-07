"""Smoke-test an installed distribution from outside the source directory."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from modelatlas import __version__, fit_som, plot_som, save_figure


def main() -> None:
    assert version("modelatlas-viz") == __version__
    x = np.random.default_rng(7).normal(size=(40, 3))
    som = fit_som(x, (3, 4), topology="hexagonal", steps=75, seed=11)
    assert som.occupancy().sum() == len(x)
    fig, ax = plot_som(som, color_by="u_matrix")
    assert ax.figure is fig
    with TemporaryDirectory() as temporary:
        svg, pdf, png = save_figure(fig, Path(temporary) / "hex_som")
        assert __version__ in svg.read_text(encoding="utf-8")
        assert pdf.read_bytes().startswith(b"%PDF")
        assert png.read_bytes().startswith(b"\x89PNG")
    print(f"ModelAtlas {__version__} fresh-install hex SOM and SVG/PDF/PNG smoke test passed")


if __name__ == "__main__":
    main()
