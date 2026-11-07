"""Compact oblique architecture diagrams for adaptive prototype graphs.

All coordinates and connections originate from a fitted prototype graph.
The oblique prism extrusion is *decorative only*; graph placement comes from
PCA of the codebook and its third principal component, and edges are learned.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def _color_mul(c, factor):
    return tuple(np.clip(np.array(colors.to_rgb(c))*factor,0,1))


def _prism(ax, point, width, fill, face_edge='#213B50', height=.115, zorder=10):
    """Oblique 3D-looking tile, constant decorative depth for all nodes."""
    x,y=point
    w=width
    dx=0.38*w;dy=0.24*w
    t=np.array([[x-.66*w,y-.13*w],[x-.14*w,y+.59*w],
                [x+.70*w,y+.26*w],[x+.18*w,y-.46*w]])
    e=np.array([0,-height])
    # shaded faces first, top surface last
    for ids,fc in [([2,3],.69),([3,0],.78)]:
        a,b=t[ids]
        poly=np.array([a,b,b+e,a+e])
        ax.add_patch(Polygon(poly,closed=True,facecolor=_color_mul(fill,fc),
                             edgecolor=face_edge,linewidth=.35,zorder=zorder))
    ax.add_patch(Polygon(t,closed=True,facecolor=fill,edgecolor=face_edge,
                         linewidth=.55,zorder=zorder+.2))
    ax.add_patch(Circle((x+.01*w,y+.02*w),radius=.080*w,facecolor='#152F45',
                        edgecolor='none',zorder=zorder+.3))


def plot_graph_atlas(graph, assignments, spectroscopic_z, *,
                     feature_labels=None, show_inputs=True, max_input_targets=11,
                     figsize=(6.8,6.0), output=None):
    """Render fixed, fitted graph as an oblique compact scientific plot.

    Graph layout = first 3 PCs of *its prototypes* (not a native lattice).
    Topological edges = fitted GNG adjacency.
    Prism hue = usable-spectroscopy fraction within assigned target objects.
    Prism width = occupancy, independently normalized.
    Fine input-to-prism lines = normalized prototype feature coordinates,
    shown only for selected prototypes; not fitted MLP layer weights.
    Prism extrusion = constant, aesthetic; NOT a fourth latent dimension.
    """
    from sklearn.decomposition import PCA
    from .adaptive import PrototypeGraph
    if not isinstance(graph, PrototypeGraph):
        raise TypeError('graph must be PrototypeGraph')
    w=np.asarray(graph.prototypes,float)
    ids=np.asarray(assignments,int)
    z=np.asarray(spectroscopic_z,float)
    if ids.shape!=z.shape or np.any(ids<0) or np.any(ids>=len(w)):
        raise ValueError('bad assignments or redshifts')
    n=len(w)
    counts=np.bincount(ids,minlength=n)
    spec=np.bincount(ids[np.isfinite(z)],minlength=n)
    cov=np.divide(spec,counts,out=np.zeros(n,float),where=counts>0)
    pca=PCA(n_components=3).fit(w)
    pc=pca.transform(w)
    pc/=np.std(pc,axis=0,ddof=1).clip(.1)
    # Orthographic projection from 3 codebook PCs to the drawing plane.
    # This is a layout, not a claim that photo-z lives in three dimensions.
    XY=np.column_stack([1.05*pc[:,0]+.55*pc[:,1]-.31*pc[:,2],
                        -.28*pc[:,0]+.67*pc[:,1]+.54*pc[:,2]])
    XY*=1.05
    # Keep left input layer separate from the graph. Zoom to content.
    xmin,xmax=XY[:,0].min(),XY[:,0].max()
    ymin,ymax=XY[:,1].min(),XY[:,1].max()
    input_x=xmin-1.43
    rangen=(ymax-ymin)
    n_features=w.shape[1]
    labels=feature_labels or [f'x{j+1}' for j in range(n_features)]
    if len(labels)!=n_features:raise ValueError('feature labels mismatch')
    node_y=np.linspace(ymax-.06*rangen,ymin+.06*rangen,n_features)
    scale=np.sqrt(counts/np.max(counts).clip(1))
    tilewidth=.10+.105*scale
    # Transparent edges still show true learned adjacency.
    fig,ax=plt.subplots(figsize=figsize,facecolor='white')
    fig.subplots_adjust(left=.025,right=.98,bottom=.12,top=.95)
    ax.set_facecolor('white')
    ax.set_aspect('equal',adjustable='box')
    cmap=plt.get_cmap('viridis')
    vmax=max(.35,float(np.percentile(cov[counts>0],99)))
    norm=Normalize(0,vmax)
    for i,j in graph.edges:
        p,q=XY[int(i)],XY[int(j)]
        ax.plot([p[0],q[0]],[p[1],q[1]],color='#6A90AA',alpha=.25,
                linewidth=.72,zorder=2)
    # Codebook-coordinate fibers: select representative/high-occupancy prototypes.
    if show_inputs:
        candidate=np.argsort(counts)[::-1][:min(max_input_targets,len(counts))]
        standardized=(w-w.mean(axis=0))/(np.std(w,axis=0).clip(.1))
        for j in range(n_features):
            xx=input_x+.12
            yy=node_y[j]
            for i in candidate:
                val=standardized[int(i),j]
                linecolor=('#D97154' if val>=0 else '#2F73B5')
                ax.plot([xx,XY[i,0]],[yy,XY[i,1]],color=linecolor,
                        linewidth=.34+.20*min(1.6,abs(val)),alpha=.10,
                        zorder=1)
    # Node prisms drawn back-to-front for plausible occlusion.
    for i in np.argsort(XY[:,1])[::-1]:
        fill='#CFD7DF' if counts[i]==0 else cmap(norm(cov[i]))
        _prism(ax,XY[i],tilewidth[i],fill,zorder=10+(-XY[i,1]))
    if show_inputs:
        for j,lab in enumerate(labels):
            y=node_y[j]
            ax.add_patch(Circle((input_x+.12,y),radius=.062,
                                facecolor='#DBE5EF',edgecolor='#27445D',lw=.85,zorder=20))
            ax.text(input_x-.04,y,lab,va='center',ha='right',fontsize=8,
                    color='#264357')
        ax.text(input_x+.12,ymax+.25,'Photometry',fontsize=8,color='#567186',ha='center')
    ax.text(xmin+.14,ymax+.25,'Adaptive prototypes · learned graph edges',
            fontsize=9,color='#344B60',ha='left')
    ax.set_xlim(input_x-.70,xmax+.52)
    ax.set_ylim(ymin-.28,ymax+.48)
    ax.axis('off')
    cbax=fig.add_axes([.31,.07,.39,.017])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbax,orientation='horizontal')
    cb.ax.tick_params(labelsize=7,length=2)
    cb.set_label('Usable spectroscopic fraction',fontsize=8,labelpad=4,color='#4C6378')
    if output is not None:
        from pathlib import Path
        out=Path(output)
        for ext in ('png','svg','pdf'):
            fig.savefig(out.with_suffix('.'+ext),dpi=250,bbox_inches='tight',pad_inches=.05)
    return fig,ax,{'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
                   'n_graph_edges':len(graph.edges),'displayed_input_target_count':int(min(max_input_targets,n)),
                   'max_coverage_colorbar':vmax}