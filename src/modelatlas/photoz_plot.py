"""Data-driven oblique 3D-looking lattice maps for photo-z cell diagnostics.

Only map topology and scalar cell data are encoded. The third dimension is a
fixed extrusion for visual legibility; it is NOT latent physical dimension.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Polygon, Circle
from .photoz import PhotoZBounds

_METRICS = {
  'target_occupancy': ('Target galaxies / cell', 'viridis'),
  'spectroscopic_coverage': ('Usable z / target', 'cividis'),
  'attempt_fraction': ('Spectroscopic attempts / target', 'cividis'),
  'success_given_attempt': ('Usable z / attempts', 'cividis'),
  'identified_width': ('Width of identified fraction', 'magma'),
  'population_width_contribution': ('Contribution to population bound width', 'magma'),
  'lower_bound': ('Lower bound for P(z > z₀)', 'viridis'),
  'upper_bound': ('Upper bound for P(z > z₀)', 'viridis'),
  'observed_high_fraction': ('P(z > z₀) among usable z', 'coolwarm'),
}


def plot_cell_plane(values, *, label='', cmap='viridis', value_range=None,
                    depth=.48, elevation='constant', figsize=(4.15,4.20), highlight=None,
                    hatch_mask=None, hatch_label=None, label_ticks=None, ax=None):
    """Compact, orthographic vector rendering of any 2D SOM scalar diagnostic.

    (fig, ax), usable in PDF/SVG/PNG via fig.savefig. The dimension extruded
    from a two-dimensional lattice is decorative, not a data dimension.

    values: finite floats or nan (unpopulated); (row, col) lattice.
    hatch_mask: boolean 2D mask, e.g. nonempty cells with zero usable z.
    value_range: explicit [min,max] or None for data min/max.
    """
    value=np.asarray(values,float)
    if value.ndim!=2 or min(value.shape)<2 or np.isinf(value).any():
        raise ValueError('values must be a 2D lattice with finite values or NaNs')
    if not np.any(np.isfinite(value)):
        raise ValueError('at least one populated cell with finite value is required')
    rows,cols=value.shape
    if hatch_mask is not None:
        h=np.asarray(hatch_mask,bool)
        if h.shape!=value.shape:raise ValueError('hatch_mask dimension mismatch')
    else:h=np.zeros_like(value,dtype=bool)
    if not (.10<=depth<=.85):raise ValueError('depth must be between 0.10 and 0.85')
    if elevation not in ('constant','metric'):
        raise ValueError('elevation must be constant or metric')
    if highlight is not None and not (0<=highlight[0]<rows and 0<=highlight[1]<cols):
        raise ValueError('highlight must be a (row,col) position within SOM')
    if value_range is None:
        low=float(np.nanmin(value));high=float(np.nanmax(value))
        if high<=low:high=low+1
    else:
        low,high=map(float,value_range)
        if not np.isfinite([low,high]).all() or low>=high:
            raise ValueError('value_range must be finite increasing pair')
    norm=Normalize(low,high)
    colormap=colormaps[cmap]
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize,dpi=240)
        fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    else:
        fig=ax.figure
    fig.patch.set_facecolor('white');ax.set_facecolor('white')
    ax.set_axis_off();ax.set_aspect('equal');ax.set_xlim(0,1.56);ax.set_ylim(0,1.38)
    O=np.array([.205,1.21])
    vr=np.array([.050,-.076])*8/(rows-1)
    vc=np.array([.087,-.031])*8/(cols-1)
    down=np.array([.026,.348])*depth
    s=.467
    def p(r,c):return O + r*vr + c*vc
    outline=np.array([p(-s,-s),p(-s,cols-1+s),p(rows-1+s,cols-1+s),p(rows-1+s,-s)])
    ax.add_patch(Polygon(outline+np.array([.018,-.038])-down,
                         facecolor='#283650',alpha=.065,edgecolor='none',zorder=-2))
    ordering=sorted(((i,j) for i in range(rows) for j in range(cols)),key=lambda ij:(ij[0]+ij[1],ij[0]))
    for i,j in ordering:
        middle=p(i,j)
        q=np.array([p(i-s,j-s),p(i-s,j+s),p(i+s,j+s),p(i+s,j-s)])
        v=value[i,j]
        if elevation=='metric':
            # Semantically meaningful relief: vertical lift encodes the *same*
            # scalar cell metric as hue, not an unmeasured 3rd latent axis.
            scaled=np.clip(norm(v),0,1) if np.isfinite(v) else 0.
            lift=np.array([.017,.15])*scaled
            top=q+lift
        else:
            top=q
        below=q-down
        if np.isfinite(v):base=np.asarray(colormap(norm(v))[:3])
        else:base=np.array([.91,.92,.94])
        def shade(mult): return tuple(np.clip(base*mult,0,1))
        zorder=10+.015*(i+j)
        ax.add_patch(Polygon([top[1],top[2],below[2],below[1]],
            facecolor=shade(.58),edgecolor='#2D3646',linewidth=.38,zorder=zorder))
        ax.add_patch(Polygon([top[2],top[3],below[3],below[2]],
            facecolor=shade(.76),edgecolor='#2D3646',linewidth=.38,zorder=zorder+.001))
        ax.add_patch(Polygon(top,facecolor=base,edgecolor='#243144',
                             linewidth=.37,zorder=zorder+.002,joinstyle='miter'))
        if h[i,j]:
            ax.add_patch(Polygon(top,facecolor='none',edgecolor='#243144',
                                 hatch='////',linewidth=.23,zorder=zorder+.003))
        if highlight is not None and (i,j)==tuple(highlight):
            # A compact marker, not a selected polygon that competes with the
            # neuron geometry. Identical location can be tracked across maps.
            ax.add_patch(Circle(middle + (lift if elevation=='metric' else 0),.020,facecolor='white',edgecolor='#26354C',
                                linewidth=1.45,zorder=41))
            ax.add_patch(Circle(middle + (lift if elevation=='metric' else 0),.006,facecolor='#26354C',edgecolor='none',zorder=42))
    # Colorbar lives inside the plot object; no overlaid slide header or footer.
    xa,xb=.34,1.28; yb=.115; hh=.018
    for k in range(160):
        aa=k/160;bb=(k+1)/160
        ax.add_patch(Polygon([(xa+(xb-xa)*aa,yb),(xa+(xb-xa)*bb,yb),
                              (xa+(xb-xa)*bb,yb+hh),(xa+(xb-xa)*aa,yb+hh)],
                 facecolor=colormap((aa+bb)/2),edgecolor='none',zorder=70))
    ax.text((xa+xb)/2,.159,label,fontsize=9.5,ha='center',va='bottom',color='#374255')
    fmt=(lambda v: f'{v:,.0f}' if abs(v)>=10 or np.isclose(v,0) else f'{v:.2f}')
    ax.text(xa,.103,fmt(low),ha='left',va='top',fontsize=8.4,color='#516078')
    ax.text(xb,.103,fmt(high),ha='right',va='top',fontsize=8.4,color='#516078')
    if hatch_label and np.any(h) and elevation=='constant':
        ax.text((xa+xb)/2,.038,hatch_label,ha='center',va='bottom',
                color='#677287',fontsize=7.8)
    return fig,ax


def plot_photoz_map(bounds:PhotoZBounds, metric='identified_width', **kwargs):
    """Render a photo-z partial-identification diagnostic on the SOM lattice."""
    if not isinstance(bounds,PhotoZBounds): raise TypeError('expected PhotoZBounds')
    if metric not in _METRICS: raise ValueError('unsupported photo-z metric: '+metric)
    label,cmap=_METRICS[metric]
    defaults={'cmap':cmap,'label':label}
    if metric in ('identified_width','lower_bound','upper_bound',
                  'spectroscopic_coverage','attempt_fraction','success_given_attempt',
                  'observed_high_fraction'):
        defaults['value_range']=(0.,1.)
    if metric in ('identified_width','lower_bound','upper_bound',
                  'population_width_contribution'):
        defaults['hatch_mask']=bounds.unidentified
        defaults['hatch_label']='Hatched: no usable spectroscopic z'
    defaults.update(kwargs)
    return plot_cell_plane(bounds.metric(metric),**defaults)


def plot_photoz_diagnostics(bounds:PhotoZBounds, *,
    metrics=('target_occupancy','spectroscopic_coverage',
             'observed_high_fraction','identified_width'),
    highlight=None, figsize=(8.6,8.6)):
    """Publication-style square small multiples: one common SOM topology.

    Not a slide: each 3D lattice and legend is an ordinary Matplotlib Axes,
    vector-native on SVG/PDF export. Explicitly no model-agnostic assumption
    that the 3D surface represents a measured manifold.
    """
    if len(metrics)!=4:
        raise ValueError('four cell metrics required for a balanced 2x2 plot')
    fig,axs=plt.subplots(2,2,figsize=figsize,dpi=210)
    fig.subplots_adjust(left=.012,right=.988,top=.99,bottom=.01,wspace=.015,hspace=.04)
    for ax,metric in zip(axs.ravel(),metrics):
        plot_photoz_map(bounds,metric=metric,highlight=highlight,ax=ax)
    return fig,axs
