"""Scientific SOM input-to-prototype architecture: compact vector-native plot.

This drawing shows the genuine dense bipartite parameterization of a fitted
SOM. There is one input *feature* node for each dimension and one colored
output prism for each codebook vector; a segment (k, i, j) corresponds to the
actual learned prototype coordinate w[i, j, k]. The SOM lattice is imposed
2-D topology. Prism thickness is a rendering cue, not a learned third axis.

No sklearn/neural-network class is assumed; the call works on any SOMMap or
finite codebook of shape (rows, columns, features).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Polygon, Circle

from .som import SOMMap

FEATURE_COLORS = ('#2457C5', '#DF5348', '#119483', '#9364B2', '#D3A02D',
                  '#3A95AF', '#B65E80', '#849142')


def _shade(colour, factor):
    return tuple(np.clip(np.asarray(to_rgb(colour))*factor, 0., 1.))


def plot_som_architecture(
    som_or_codebook,
    *,
    values=None,
    metric_label=None,
    feature_labels=None,
    cmap='turbo',
    value_range=None,
    highlight=None,
    show_connections='all',
    emphasize_highlight=True,
    show_neuron_centers=True,
    show_colorbar=True,
    depth=.40,
    figsize=(5.30, 4.70),
    hatch_mask=None,
):
    """Draw true input nodes, SOM prototype connections, and a tilted 3D map.

    Parameters
    ----------
    som_or_codebook : SOMMap or array (rows,cols,features)
        *Fitted* standardized SOM codebook. Each line is one actual w_ijk.
    values : (rows,cols) array, optional
        Measured/derived quantity to encode as prism color, e.g. photo-z
        identified width. Defaults to codebook component 0.
    metric_label : str, optional
        Quantitative colorbar label, never inferred from model weights.
    feature_labels : iterable of str, optional
        Labels for input feature nodes. Defaults to SOMMap.feature_labels.
    value_range : (lo,hi), optional
        Lock color normalization across comparable figures.
    highlight : (row,col), optional
        Output prototype to outline and connect prominently.
    show_connections : {'all','selected','none'}
        'all' draws n_features * n_neurons real links; 'selected' only the
        highlighted prototype's links, and 'none' draws no links.
    emphasize_highlight : bool
        Enlarge color-coded links to the highlighted prototype. Color identifies
        the feature; line thickness reflects absolute standardized component.
        There is no implied causal or predictive feature importance.
    show_neuron_centers : bool
        Mark actual output SOM neurons with small center dots on their prisms.
    depth : float
        Constant visual extrusion in projected units; *not* an inferred axis.
    hatch_mask : (rows,cols) bool array, optional
        Mark cells with no observed spectroscopy using diagonal hatch.
    
    Returns
    -------
    (fig, ax) : Matplotlib Figure and Axes for PNG, SVG or PDF export.
    """
    M = som_or_codebook if isinstance(som_or_codebook,SOMMap) else SOMMap(np.asarray(som_or_codebook))
    rows, cols, nfeatures=M.codebook.shape
    if nfeatures > 8:
        raise ValueError('More than 8 input nodes is too dense for this view; select or group input features explicitly')
    if nfeatures < 1:
        raise ValueError('codebook has no input features')
    if show_connections not in ('all','selected','none'):
        raise ValueError("show_connections must be 'all', 'selected', or 'none'")
    if show_connections == 'selected' and highlight is None:
        raise ValueError("show_connections='selected' requires a highlight neuron")
    if highlight is not None:
        if len(highlight)!=2 or not (0<=highlight[0]<rows and 0<=highlight[1]<cols):
            raise ValueError('highlight must be a valid (row,col)')
        highlight=tuple(map(int, highlight))
    if not (.1<=depth<=.9):
        raise ValueError('depth must be between .1 and .9')
    if feature_labels is None:
        feature_labels=M.feature_labels or tuple(f'$x_{{{j+1}}}$' for j in range(nfeatures))
    if len(feature_labels)!=nfeatures:
        raise ValueError('one label per feature is required')

    if values is None:
        values=M.display_codebook[...,0] if M.display_codebook is not None else M.codebook[...,0]
        metric_label=metric_label or str(feature_labels[0]) + ' · prototype value'
    else:
        values=np.asarray(values, dtype=float)
        metric_label=metric_label or 'Cell metric'
    if values.shape != (rows,cols) or np.isinf(values).any() or not np.any(np.isfinite(values)):
        raise ValueError('values must match the SOM lattice and contain finite data')
    if hatch_mask is not None:
        hatch_mask=np.asarray(hatch_mask,bool)
        if hatch_mask.shape != (rows,cols):
            raise ValueError('hatch_mask must match the SOM lattice')

    lo=float(np.nanmin(values)) if value_range is None else float(value_range[0])
    hi=float(np.nanmax(values)) if value_range is None else float(value_range[1])
    if not np.isfinite([lo,hi]).all() or lo>hi:
        raise ValueError('value range invalid')
    if np.isclose(lo,hi): hi=lo+1
    norm=Normalize(lo,hi,clip=True)
    cm=colormaps[cmap]

    fig,ax=plt.subplots(figsize=figsize,dpi=240)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    fig.subplots_adjust(left=.008,right=.996,top=.995,bottom=.004)
    ax.set_axis_off()
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-.01,1.55); ax.set_ylim(.005,1.17)

    # One axonometric projection shared by codebook, neurons and input-to-unit
    # links. No mpl_toolkits.mplot3d painter artefacts, no raster layers.
    base=np.array([.560,1.080])
    vrow=np.array([.041,-.073]) * 8/max(rows-1,1)
    vcol=np.array([.065,-.026]) * 8/max(cols-1,1)
    vextr=np.array([.020,.205]) * depth
    gap=.064
    half=(1-gap)/2
    def project(r,c):return base+float(r)*vrow+float(c)*vcol

    # Left input nodes: exactly one node per fitted feature.
    nx=.184
    ny=np.linspace(.826,.246,nfeatures) if nfeatures>1 else np.array([.51])

    # Every faint strand is grounded in the real prototype matrix. For SOMs
    # the codebook is a collection of fitted prototype vectors, not the same
    # as a sequence of trainable interlayer matrices in a generic MLP.
    segments=[]; colors=[]; widths=[]
    if show_connections=='all':
        for k,y in enumerate(ny):
            for i in range(rows):
                for j in range(cols):
                    segments.append([(nx+.013,y),project(i,j)])
                    colors.append((*to_rgb(FEATURE_COLORS[k]),.115))
                    widths.append(.34)
        ax.add_collection(LineCollection(segments,colors=colors,linewidths=widths,
                         zorder=1,capstyle='round', antialiaseds=True))

    if highlight is not None and show_connections!='none' and emphasize_highlight:
        hi_,hj=highlight
        raw=M.codebook[hi_,hj,:]
        # Normalize within feature across prototypes; do not confound numerical
        # units, especially g-r vs i magnitude. Used for *visibility* only.
        denom=np.percentile(np.abs(M.codebook),90,axis=(0,1))
        strength=np.clip(np.abs(raw)/np.maximum(denom,1e-8),0,1)
        for k,y in enumerate(ny):
            # Emphasized selected links appear before prisms: correct occlusion.
            a=.42+.30*strength[k]
            ax.plot([nx+.018,project(hi_,hj)[0]], [y,project(hi_,hj)[1]],
                    color=(*to_rgb(FEATURE_COLORS[k]),a),
                    lw=1.05+1.12*strength[k], zorder=2, solid_capstyle='round')

    # Soft shadow beneath lattice, excluding any disconnected presentation
    # elements. Shadow is a depth cue, not measured data.
    polygon=np.array([project(-half,-half),project(-half,cols-1+half),
             project(rows-1+half,cols-1+half),project(rows-1+half,-half)])
    ax.add_patch(Polygon(polygon-vextr+(.020,-.042),closed=True,
                 facecolor='#1A2540',edgecolor='none',alpha=.055,zorder=3))

    draw_order=sorted(((i,j) for i in range(rows) for j in range(cols)),
                      key=lambda ij:(ij[0]+ij[1],ij[0]))
    neuron_centers=[]
    for i,j in draw_order:
        p=np.array([project(i-half,j-half), project(i-half,j+half),
                    project(i+half,j+half), project(i+half,j-half)])
        bottom=p-vextr
        if np.isfinite(values[i,j]):
            color=cm(norm(values[i,j]))
        else:
            color=('#D0D5DE')
        z=10+.028*(i+j)+.0002*i
        ax.add_patch(Polygon([p[1],p[2],bottom[2],bottom[1]],
                     facecolor=_shade(color,.55),edgecolor='#25334A',
                     lw=.28,zorder=z))
        ax.add_patch(Polygon([p[2],p[3],bottom[3],bottom[2]],
                     facecolor=_shade(color,.78),edgecolor='#25334A',
                     lw=.28,zorder=z+.001))
        ax.add_patch(Polygon(p,facecolor=color,
                     edgecolor='#19324A',lw=.34,zorder=z+.002,joinstyle='miter'))
        if hatch_mask is not None and hatch_mask[i,j]:
            ax.add_patch(Polygon(p,facecolor='none',edgecolor='#25334A',
                         lw=.15,hatch='////',zorder=z+.003))
        if show_neuron_centers:
            # A small visible mark centered on every output prototype tile.
            center=project(i,j)
            neuron_centers.append(center)
            ax.add_patch(Circle(center,.0042,facecolor='#0C253D',
                         edgecolor='none',alpha=.70,zorder=z+.004))
        if highlight == (i,j):
            ax.add_patch(Polygon(p,facecolor='none',edgecolor='white',
                         lw=1.85,zorder=z+.005))
            ax.add_patch(Circle(project(i,j),.016,facecolor='white',
                         edgecolor='#162238',lw=1.30,zorder=z+.006))
            ax.add_patch(Circle(project(i,j),.0042,facecolor='#14223A',
                         edgecolor='none',zorder=z+.007))

    for k,y in enumerate(ny):
        col=FEATURE_COLORS[k]
        ax.add_patch(Circle((nx,y),.0188,facecolor=col,edgecolor='white',
                     lw=1.45,zorder=60))
        ax.text(nx-.031,y,str(feature_labels[k]),fontsize=10.2,color='#2A3447',
                ha='right',va='center',zorder=61)

    if show_colorbar:
        x0,x1=.84,1.50
        y0,y1=.084,.102
        for n in range(115):
            a=n/115;b=(n+1)/115
            ax.add_patch(Polygon([(x0+(x1-x0)*a,y0),(x0+(x1-x0)*b,y0),
                         (x0+(x1-x0)*b,y1),(x0+(x1-x0)*a,y1)],
                         facecolor=cm((a+b)/2),edgecolor='none',zorder=70))
        ax.text((x0+x1)/2,.125,metric_label,fontsize=9.2,
                color='#3B4556',ha='center',va='bottom',zorder=70)
        f=lambda v: f'{v:.0f}' if abs(v)>=10 else f'{v:.2f}'
        ax.text(x0,.073,f(lo),fontsize=8.5,color='#596578',ha='left',va='top')
        ax.text(x1,.073,f(hi),fontsize=8.5,color='#596578',ha='right',va='top')

    # Testable metadata—one segment per real codebook component.
    ax._modelatlas_architecture={
        'features':nfeatures, 'neurons':rows*cols,
        'connections':len(segments), 'highlighted_connections':
        (nfeatures if highlight is not None and show_connections!='none' and emphasize_highlight else 0),
        'highlight':highlight, 'units': 'standardized codebook components',
        'colored_metric':metric_label,
    }
    return fig,ax


def plot_photoz_architecture(bounds, som, *, metric='identified_width', **kwargs):
    """Use photo-z identified cell metrics to color the fitted SOM network.

    Requires the SOM separately because PhotoZBounds deliberately contains
    identification results only, not the model's fitted codebook.
    """
    from .photoz import PhotoZBounds
    from .photoz_plot import _METRICS
    if not isinstance(bounds,PhotoZBounds):
        raise TypeError('bounds must be a PhotoZBounds')
    if metric not in _METRICS:
        raise ValueError('unknown photo-z metric')
    label,_=_METRICS[metric]
    m=som if isinstance(som,SOMMap) else SOMMap(som)
    if m.codebook.shape[:2]!=bounds.n_target.shape:
        raise ValueError('som and bounds must have matching lattice')
    defaults={'metric_label':label, 'value_range':(0.,1.),
              'hatch_mask':bounds.unidentified}
    if metric in ('target_occupancy','population_width_contribution'):
        defaults.pop('value_range')
    defaults.update(kwargs)
    return plot_som_architecture(m,values=bounds.metric(metric),**defaults)
