"""ModelAtlas: model structure and inferential geometry (research preview)."""
from .som import SOMMap, train_som, som_from_minisom
from .render import plot_som, plot_metric
from .regression import plot_coefficients, plot_predictions
from .network import plot_mlp
from .photoz import PhotoZCells, PhotoZBounds, sharp_tail_bounds
from .photoz_plot import plot_photoz_map, plot_cell_plane, plot_photoz_diagnostics
from .architecture import plot_som_architecture, plot_photoz_architecture
from .api import visualize
__all__=["SOMMap","train_som","som_from_minisom","plot_som","plot_metric",
         "plot_coefficients","plot_predictions","plot_mlp","visualize",
         "PhotoZCells","PhotoZBounds","sharp_tail_bounds","plot_photoz_map","plot_cell_plane","plot_photoz_diagnostics","plot_som_architecture","plot_photoz_architecture"]
__version__="0.5.0"

# Optional XAI extras: import separately when scipy is unavailable.
from .explanation import PathExplanation, explain_mlp, activation_signatures
from .explanation_plot import plot_explanation_atlas

from .adaptive import PrototypeGraph, GTMModel, growing_neural_gas, train_gtm, partition_tail_bounds
from .adaptive_plot import plot_graph_atlas
__all__.extend(["PrototypeGraph", "GTMModel", "growing_neural_gas", "train_gtm", "partition_tail_bounds", "plot_graph_atlas"])
