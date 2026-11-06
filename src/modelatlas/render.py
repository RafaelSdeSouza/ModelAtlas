"""Matplotlib-only, vector-native oblique SOM prism renderer.

No mplot3d: deterministic orthographic projection permits precise drawing
order, true vector paths, and consistent export to SVG/PDF/PNG.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba, to_rgb
from matplotlib.patches import Polygon, Circle
from matplotlib import colormaps
from .som import SOMMap

_NODE_COLORS=("#2654C8", "#D95543", "#15887A", "#9661A9", "#CD9A32", "#3F8EC8", "#C16378")

def _shade(col, factor):
    rgb=np.array(to_rgb(col))
    return tuple(np.clip(rgb*factor,0,1))


def _metric(map_, color_by):
    if isinstance(color_by,int):
        if color_by<0 or color_by>=map_.n_features: raise ValueError("color feature out of range")
        a=map_.codebook if map_.display_codebook is None else map_.display_codebook
        label=(map_.feature_labels[color_by] if map_.feature_labels else f"feature {color_by+1}")
        if map_.feature_units:
            label+=f" ({map_.feature_units[color_by]})" if map_.feature_units[color_by] else ""
        return a[...,color_by],label
    if color_by=="occupancy": return map_.occupancy(),"observations per neuron"
    if color_by=="u_matrix": return map_.u_matrix(),"neighbor distance (scaled features)"
    raise ValueError("color_by must be a feature index, 'occupancy', or 'u_matrix'")


def plot_som(map_or_weights, *, color_by=0, labels=None, highlight=None,
             depth=.36, elevation="constant", connections=True, colorbar=True,
             cmap="turbo", figsize=(5.0,4.8), azimuth=1.0):
    """Draw an oblique 3-D-looking SOM as a single compact Matplotlib axes.

    The grid is a 2D SOM, not an inferred three-dimensional manifold. Top-face
    color carries the selected observed parameter, occupancy or U-distance.
    If `elevation='occupancy'`, the prism thickness also carries occupancy;
    otherwise depth is decorative and equal for every neuron.

    Parameters
    ----------
    map_or_weights : SOMMap or ndarray(rows, cols, features)
    color_by : int | 'occupancy' | 'u_matrix'
    highlight : (row, col) | None  (e.g., best-matching unit of an observation)
    elevation : 'constant' | 'occupancy'
    azimuth : float  factor controlling obliqueness, from ~0.65 to ~1.25

    Returns a real (Figure, Axes). Save using fig.savefig(..., transparent=True).
    """
    M=map_or_weights if isinstance(map_or_weights,SOMMap) else SOMMap(map_or_weights)
    rows,cols,nf=M.codebook.shape
    if not (0.04<=depth<=1.0): raise ValueError("depth must be in [0.04,1]")
    if elevation not in ("constant","occupancy"): raise ValueError("bad elevation")
    if not (.5<=azimuth<=1.4): raise ValueError("azimuth must be [0.5,1.4]")
    if highlight is not None and (not 0<=highlight[0]<rows or not 0<=highlight[1]<cols):
        raise ValueError("highlight outside lattice")
    val,legend=_metric(M,color_by)
    val=np.asarray(val,float)
    mn,mx=float(val.min()),float(val.max())
    norm=Normalize(mn,mx if mx>mn else mn+1)
    cm=colormaps[cmap]
    if labels is None:
        labels=M.feature_labels or tuple(f"x{i+1}" for i in range(nf))
    if len(labels)!=nf: raise ValueError("one label per feature required")
    if elevation=="occupancy":
        occup=M.occupancy()
        # height has explicit interpretation, scaled to [0.20, 0.65] world units.
        heights=depth*(.26+.74*np.sqrt(occup/max(1,occup.max())))
    else: heights=np.full((rows,cols),depth)

    fig,ax=plt.subplots(figsize=figsize,dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=.005,right=.995,top=.995,bottom=.005)
    ax.set_axis_off();ax.set_aspect("equal")
    ax.set_xlim(0,1.40);ax.set_ylim(.035,1.245)

    origin=np.array([.438,1.145])
    vcol=np.array([.085,-.041*azimuth])*7/max(cols-1,1)
    vrow=np.array([.043,-.090*azimuth])*6/max(rows-1,1)
    vdepth=np.array([.022,.191])
    s=.462
    def project(r,c,z=0): return origin+r*vrow+c*vcol+z*vdepth

    # Ground shadow: small, visually sufficient to anchor the tilted lattice.
    ext=np.array([project(-s,-s),project(-s,cols-1+s),
                  project(rows-1+s,cols-1+s),project(rows-1+s,-s)])
    ax.add_patch(Polygon(ext+np.array([.018,-.042]),closed=True,
                         facecolor="#09285B",alpha=.052,edgecolor="none",zorder=-4))

    # SOM codebook entries are prototypes, not causal feature importances.
    # Draw input-to-prototype strands with constant (very low) opacity.
    yinputs=np.linspace(.845,.310,nf) if nf>1 else np.array([.57])
    xinput=.132
    if connections:
        segments=[];colors=[]
        for k,y in enumerate(yinputs):
            for i in range(rows):
                for j in range(cols):
                    segments.append([(xinput+.014,y),project(i,j,-.045)])
                    colors.append((*to_rgb(_NODE_COLORS[k%len(_NODE_COLORS)]),.068))
        ax.add_collection(LineCollection(segments,colors=colors,linewidths=.31,
                                         antialiaseds=True,zorder=0))

    order=sorted(((i,j) for i in range(rows) for j in range(cols)),
                 key=lambda ij:(ij[0]+ij[1],ij[0]))
    for i,j in order:
        p=np.array([project(i-s,j-s),project(i-s,j+s),
                    project(i+s,j+s),project(i+s,j-s)])
        below=p-vdepth*heights[i,j]
        base=cm(norm(val[i,j]))
        z=20+.019*(i+j)
        side=Polygon([p[1],p[2],below[2],below[1]],closed=True,
                     facecolor=_shade(base,.56),edgecolor="#25314A",
                     linewidth=.29,zorder=z)
        front=Polygon([p[2],p[3],below[3],below[2]],closed=True,
                      facecolor=_shade(base,.76),edgecolor="#25314A",
                      linewidth=.29,zorder=z+.0001)
        cap=Polygon(p,closed=True,facecolor=base,edgecolor="#112747",
                    linewidth=.3,zorder=z+.0002,joinstyle="miter")
        ax.add_patch(side);ax.add_patch(front);ax.add_patch(cap)
        if highlight is not None and (i,j)==tuple(highlight):
            ax.add_patch(Polygon(p,closed=True,fill=False,edgecolor="white",
                                 linewidth=1.9,zorder=50))
            ax.add_patch(Circle(project(i,j,.02),.012,facecolor="white",
                                edgecolor="#1B2439",lw=.85,zorder=51))

    for k,y in enumerate(yinputs):
        ax.add_patch(Circle((xinput,y),.0175,facecolor=_NODE_COLORS[k%len(_NODE_COLORS)],
                            edgecolor="white",linewidth=1.15,zorder=80))
        ax.text(xinput-.028,y,labels[k],va="center",ha="right",fontsize=10.3,
                color="#2F3B4C",fontfamily="DejaVu Sans",zorder=81)

    if colorbar:
        x0,x1=.79,1.255
        y0,y1=.095,.111
        for k in range(88):
            a=k/88;b=(k+1)/88
            ax.add_patch(Polygon([(x0+(x1-x0)*a,y0),(x0+(x1-x0)*b,y0),
                                  (x0+(x1-x0)*b,y1),(x0+(x1-x0)*a,y1)],
                                  closed=True,facecolor=cm((a+b)/2),edgecolor="none",zorder=70))
        ax.text((x0+x1)/2,.140,legend,color="#334157",fontsize=8.3,
                ha="center",va="bottom")
        fmt=lambda x: f"{x:.0f}" if abs(x)>=20 or np.isclose(x,0) else f"{x:.2f}"
        ax.text(x0,.078,fmt(mn),fontsize=7.6,color="#59657A",ha="left",va="top")
        ax.text(x1,.078,fmt(mx),fontsize=7.6,color="#59657A",ha="right",va="top")
        if elevation=="occupancy":
            ax.text((x0+x1)/2,.040,"prism height ∝ √(occupancy)",fontsize=7.3,
                    color="#687389",ha="center",va="bottom")
    return fig,ax


def plot_metric(map_, metric="u_matrix", *, cmap="magma", figsize=(3.25,3.15),
                cell_labels=False):
    """Plot an honest top-down component plane, U-matrix, or occupancy map."""
    M=map_ if isinstance(map_,SOMMap) else SOMMap(map_)
    values,label=_metric(M,metric)
    fig,ax=plt.subplots(figsize=figsize,dpi=200)
    fig.patch.set_facecolor("white")
    im=ax.imshow(values,cmap=cmap,origin="upper",interpolation="nearest")
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xlabel(label,fontsize=9,color="#303B4D",labelpad=9)
    if cell_labels:
        for r in range(M.rows):
            for c in range(M.cols):
                ax.text(c,r,str(int(values[r,c])) if metric=="occupancy" else f"{values[r,c]:.1f}",
                        color="white" if im.norm(values[r,c])<.4 else "black",
                        fontsize=7,ha="center",va="center")
    cb=fig.colorbar(im,ax=ax,fraction=.048,pad=.025)
    cb.ax.tick_params(labelsize=7.5,length=0)
    cb.outline.set_visible(False)
    fig.tight_layout(pad=.4)
    return fig,ax
