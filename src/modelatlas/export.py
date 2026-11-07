"""Reproducible Matplotlib export and software provenance."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

from ._version import __version__


def save_figure(
    figure: Figure,
    stem: str | Path,
    *,
    formats: Iterable[str] = ("svg", "pdf", "png"),
    dpi: int = 240,
    transparent: bool = False,
) -> tuple[Path, ...]:
    """Save a figure with the ModelAtlas version in file metadata.

    The stem is a filename without an extension. In a fixed numerical,
    Matplotlib and font environment, SVG IDs and timestamps are reproducible.
    PDF and SVG preserve vector artists and selectable text where supported.
    """
    path = Path(stem)
    extensions = tuple(formats)
    if not extensions or any(ext not in ("svg", "pdf", "png") for ext in extensions):
        raise ValueError("formats must contain only svg, pdf or png")
    path.parent.mkdir(parents=True, exist_ok=True)
    software = f"ModelAtlas {__version__}"
    metadata = {
        "svg": {"Creator": software, "Date": None},
        "pdf": {"Creator": software, "CreationDate": None, "ModDate": None},
        "png": {"Software": software},
    }
    outputs = []
    with mpl.rc_context({"svg.hashsalt": software, "svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in extensions:
            target = path.with_name(f"{path.name}.{extension}")
            figure.savefig(
                target,
                format=extension,
                dpi=dpi,
                transparent=transparent,
                bbox_inches="tight",
                pad_inches=0.08,
                metadata=metadata[extension],
            )
            outputs.append(target)
    return tuple(outputs)
