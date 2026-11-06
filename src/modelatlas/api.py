"""Opt-in dispatch only for supported, semantically different model types."""
from __future__ import annotations
import numpy as np
from .som import SOMMap
from .render import plot_som
from .network import plot_mlp
from .regression import plot_coefficients
from .photoz import PhotoZBounds
from .photoz_plot import plot_photoz_map
from .architecture import plot_som_architecture, plot_photoz_architecture


def visualize(model, *, kind="auto", architecture=False, **kwargs):
    """Visualize fitted structure; never fabricate unsupported model topology.

    Dispatch priorities: SOMMap/(3D codebook) / fitted MLP / linear estimator.
    Unsupported estimators raise TypeError so the user can select an adapter.
    """
    if kind not in ("auto","som","mlp","linear","photoz"):
        raise ValueError("kind must be auto, som, mlp, linear, or photoz")
    if kind in ("auto","photoz") and isinstance(model,PhotoZBounds):
        if architecture:
            if "som" not in kwargs:
                raise ValueError("visualizing PhotoZBounds architecture requires som=the fitted SOMMap")
            som=kwargs.pop("som")
            return plot_photoz_architecture(model,som,**kwargs)
        return plot_photoz_map(model,**kwargs)
    if kind in ("auto","som") and (isinstance(model,SOMMap) or
        isinstance(model,np.ndarray) and model.ndim==3):
        return plot_som_architecture(model,**kwargs) if architecture else plot_som(model,**kwargs)
    if kind in ("auto","som") and hasattr(model,"get_weights"):
        weights=np.asarray(model.get_weights())
        if weights.ndim==3: return (plot_som_architecture(weights,**kwargs) if architecture else plot_som(weights,**kwargs))
    if architecture and kind not in ("auto","som","photoz"):
        raise ValueError("architecture=True is specific to SOM and photoz SOM rendering")
    if kind in ("auto","mlp") and hasattr(model,"coefs_"):
        return plot_mlp(model,**kwargs)
    if kind in ("auto","linear") and hasattr(model,"coef_"):
        return plot_coefficients(model,**kwargs)
    raise TypeError("No supported adapter: use SOMMap, a 3D SOM codebook, "
                    "a fitted sklearn MLP, a fitted linear estimator, or PhotoZBounds")
