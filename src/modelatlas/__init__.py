"""Vector-native scientific visualizations of fitted models."""

from ._version import __version__
from .adaptive import GTMModel, PrototypeGraph, growing_neural_gas, partition_tail_bounds, train_gtm
from .adaptive_plot import (
    PrototypeLayout,
    fit_prototype_layout,
    plot_graph_atlas,
    plot_gtm,
    plot_prototype_graph,
)
from .api import visualize
from .architecture import plot_photoz_architecture, plot_som_architecture
from .explanation import PathExplanation, activation_signatures, explain_mlp
from .explanation_plot import explanation_layout, plot_explanation_atlas
from .export import save_figure
from .geometry import ObliqueProjection, ProjectionDiagnostics, projection_diagnostics
from .lattice import SOMLattice
from .network import plot_mlp
from .photoz import PhotoZBounds, PhotoZCells, sharp_tail_bounds
from .photoz_plot import plot_cell_plane, plot_identification_sensitivity, plot_photoz_diagnostics, plot_photoz_map
from .regression import plot_coefficients, plot_predictions
from .render import plot_metric, plot_som
from .som import SOMMap, fit_som, som_from_minisom, train_som
from .symmetry import Alignment, match_tanh_mlp, representational_metrics, transform_signed_permutation
from .symmetry_plot import plot_alignment_diagnostic, plot_independent_pair, plot_residual_atlas

__all__ = [
    "Alignment",
    "GTMModel",
    "ObliqueProjection",
    "PathExplanation",
    "PhotoZBounds",
    "PhotoZCells",
    "ProjectionDiagnostics",
    "PrototypeGraph",
    "PrototypeLayout",
    "SOMLattice",
    "SOMMap",
    "__version__",
    "activation_signatures",
    "explain_mlp",
    "explanation_layout",
    "fit_prototype_layout",
    "fit_som",
    "growing_neural_gas",
    "match_tanh_mlp",
    "partition_tail_bounds",
    "plot_alignment_diagnostic",
    "plot_cell_plane",
    "plot_coefficients",
    "plot_explanation_atlas",
    "plot_graph_atlas",
    "plot_gtm",
    "plot_identification_sensitivity",
    "plot_independent_pair",
    "plot_metric",
    "plot_mlp",
    "plot_photoz_architecture",
    "plot_photoz_diagnostics",
    "plot_photoz_map",
    "plot_predictions",
    "plot_prototype_graph",
    "plot_residual_atlas",
    "plot_som",
    "plot_som_architecture",
    "projection_diagnostics",
    "representational_metrics",
    "save_figure",
    "sharp_tail_bounds",
    "som_from_minisom",
    "train_gtm",
    "train_som",
    "transform_signed_permutation",
    "visualize",
]

# Model-agnostic behavior geometry (experimental; opt in via public API).
from .functional import FunctionalMap, InformationLattice, functional_map, information_lattice, model_predictions
from .functional_plot import plot_functional_atlas, plot_information_lattice, plot_prediction_fibers
from .behavior_geometry import ProbabilityAtlas, build_probability_atlas
from .behavior_plot import plot_probability_atlas
__all__.extend(['FunctionalMap','InformationLattice','functional_map','information_lattice','model_predictions',
               'plot_functional_atlas','plot_information_lattice','plot_prediction_fibers',
               'ProbabilityAtlas','build_probability_atlas','plot_probability_atlas'])
