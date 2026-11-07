"""Configuration for the ModelAtlas documentation and executable gallery."""

import os
import sys
from pathlib import Path

from modelatlas import __version__

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))

project = "ModelAtlas"
author = "Rafael S. de Souza"
copyright = "2026, Rafael S. de Souza and ModelAtlas contributors"
version = release = __version__
root_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
]
exclude_patterns = ["_build", "figures"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["atlas.css"]
html_favicon = "_static/atlas-mark.svg"
html_title = "ModelAtlas documentation"
html_baseurl = os.environ.get("MODELATLAS_DOCS_URL", "")
html_sidebars = {"index": []}
html_context = {"default_mode": "auto"}
html_theme_options = {
    "logo": {"text": "ModelAtlas"},
    "github_url": "https://github.com/RafaelSdeSouza/ModelAtlas",
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_nav_level": 2,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
autodoc_typehints = "none"
nitpick_ignore = [("py:class", "numpy.ndarray")]
autodoc_member_order = "bysource"
add_module_names = False
linkcheck_retries = 1
linkcheck_timeout = 15


sphinx_gallery_conf = {
    "examples_dirs": ["../examples/astronomy", "../examples/gallery"],
    "gallery_dirs": ["astronomy", "gallery"],
    "copyfile_regex": r".*\.npz$",
    "filename_pattern": r"plot_",
    "within_subsection_order": "FileNameSortKey",
    "image_scrapers": ("atlas_gallery",),
    "abort_on_example_error": True,
    "run_stale_examples": True,
    "capture_repr": (),
    "download_all_examples": True,
    "show_memory": False,
    "write_computation_times": False,
}
