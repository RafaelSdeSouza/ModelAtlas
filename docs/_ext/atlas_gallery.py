"""Sphinx-Gallery image scraper with semantic figure descriptions."""

from pathlib import Path

import matplotlib.pyplot as plt
from sphinx_gallery.scrapers import figure_rst

from modelatlas import save_figure


def scrape_figures(block, block_vars, gallery_conf):
    """Record figure descriptions and publish generated vector and raster files."""
    source = Path(gallery_conf["src_dir"])
    generated = source / "_static" / "generated"
    images = []
    try:
        for number, image_path in zip(plt.get_fignums(), block_vars["image_path_iterator"]):
            fig = plt.figure(number)
            if not fig.get_label():
                raise ValueError(
                    "each gallery figure needs an accessible description via Figure.set_label"
                )
            stem = Path(image_path).with_suffix("")
            (svg,) = save_figure(fig, stem, formats=("svg",))
            save_figure(fig, generated / stem.name, formats=("svg", "pdf", "png"))
            images.append(figure_rst([svg], str(source), fig_titles=fig.get_label()))
            links = " · ".join(
                f'<a href="../_static/generated/{stem.name}.{ext}">{ext.upper()}</a>'
                for ext in ("svg", "pdf", "png")
            )
            images.append(
                f'\n.. raw:: html\n\n   <p class="atlas-downloads">Download {links}</p>\n\n'
            )
    finally:
        plt.close("all")
    return "\n".join(images)


def _get_sg_image_scraper():
    return scrape_figures
