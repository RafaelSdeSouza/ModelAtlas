"""Dispatch fitted models to their model-specific plotting functions."""

import numpy as np

from .adaptive import GTMModel, PrototypeGraph
from .adaptive_plot import plot_gtm, plot_prototype_graph
from .architecture import plot_photoz_architecture, plot_som_architecture
from .network import plot_mlp
from .photoz import PhotoZBounds
from .photoz_plot import plot_photoz_map
from .regression import plot_coefficients
from .render import plot_som
from .som import SOMMap, som_from_minisom


def visualize(model, *, kind="auto", architecture=False, **kwargs):
    """Draw a supported fitted model and return (figure, axes).

    kind can be auto, som, graph, gtm, mlp, linear or photoz. Architecture is
    an opt-in for the SOM and photo-z SOM views. Training and preprocessing
    belong to the caller.
    """
    if kind not in ("auto", "som", "graph", "gtm", "mlp", "linear", "photoz"):
        raise ValueError("kind must be auto, som, graph, gtm, mlp, linear, or photoz")
    if kind in ("auto", "photoz") and isinstance(model, PhotoZBounds):
        if architecture:
            if "som" not in kwargs:
                raise ValueError("photo-z architecture requires som=the fitted SOMMap")
            return plot_photoz_architecture(model, kwargs.pop("som"), **kwargs)
        return plot_photoz_map(model, **kwargs)
    if kind in ("auto", "som"):
        if hasattr(model, "get_weights"):
            model = som_from_minisom(model)
        if isinstance(model, SOMMap) or isinstance(model, np.ndarray) and model.ndim == 3:
            draw = plot_som_architecture if architecture else plot_som
            return draw(model, **kwargs)
    if architecture:
        raise ValueError("architecture=True is specific to SOM and photo-z SOM rendering")
    if kind in ("auto", "graph") and isinstance(model, PrototypeGraph):
        return plot_prototype_graph(model, **kwargs)
    if kind in ("auto", "gtm") and isinstance(model, GTMModel):
        return plot_gtm(model, **kwargs)
    if kind in ("auto", "mlp") and hasattr(model, "coefs_"):
        return plot_mlp(model, **kwargs)
    if kind in ("auto", "linear") and hasattr(model, "coef_"):
        return plot_coefficients(model, **kwargs)
    raise TypeError("no supported adapter for this fitted model and kind")
