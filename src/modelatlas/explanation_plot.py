"""Compact, oblique 2.5D scientific visualization of neural-network explanations.

Oblique neuron plates are a graphic arrangement, not an intrinsic SOM-like
neighborhood. Layout optimization uses correlations of activation signatures
on user-provided reference data. Edge and neuron colors encode signed path-
integrated *logit* contributions, not raw weights or causal mechanisms.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from .explanation import PathExplanation, activation_signatures

INK='#182E47'
POS='#D95D45'
NEG='#3668B6'
NEUTRAL='#F3F5F6'


def _blend(c0, c1, w):
    return tuple((1-w)*np.asarray(to_rgb(c0)) + w*np.asarray(to_rgb(c1)))


def _signed_color(value, scale):
    v=float(np.clip(value / max(scale,1e-10),-1,1))
    return _blend(NEUTRAL,POS if v>=0 else NEG,abs(v)**.72)


def _shade(c, factor):
    return tuple(np.clip(np.asarray(to_rgb(c))*factor,0,1))


def _layout_by_responses(A):
    """Map activation-response signatures to a regular grid by PCA assignment.

    Closeness is *layout-derived* from observations, not a neural topology.
    Return unit-index -> (row,col).
    """
    from scipy.optimize import linear_sum_assignment
    n=int(A.shape[1]);rows=int(np.floor(np.sqrt(n)));cols=int(np.ceil(n/rows))
    traces=A.T
    traces=traces-traces.mean(axis=1,keepdims=True)
    traces=traces/np.maximum(traces.std(axis=1,keepdims=True),1e-4)
    # SVD of unit signatures approximates pairwise activation correlation.
    centered=traces-traces.mean(axis=0,keepdims=True)
    U,s,_=np.linalg.svd(centered,full_matrices=False)
    coord=U[:,:2]*s[None,:2] if n>=3 else np.column_stack([np.arange(n),np.zeros(n)])
    coord-=coord.mean(axis=0,keepdims=True)
    coord/=np.maximum(coord.std(axis=0,keepdims=True),1e-6)
    grid=np.array([(r,c) for r in range(rows) for c in range(cols)],dtype=float)
    grid-=grid.mean(axis=0,keepdims=True)
    grid/=np.maximum(grid.std(axis=0,keepdims=True),1e-6)
    cost=np.sum((coord[:,None,:]-grid[None,:,:])**2,axis=-1)
    unit,loc=linear_sum_assignment(cost)
    mapping={int(k):tuple(map(int,np.unravel_index(int(l),(rows,cols)))) for k,l in zip(unit,loc)}
    return (rows,cols),mapping


def plot_explanation_atlas(model, explanation:PathExplanation, *, X_reference,
                           feature_names=None, ax=None, show_all_edges=True,
                           highlight=True, grid_layout=None):
    """Render a two-hidden-layer binary MLP explanation as an oblique vector plot.

    - Node fills: signed incoming edge conductance for each hidden unit.
    - Edge hue and opacity: signed edge conductance and its absolute magnitude.
    - Input nodes: signed Integrated Gradients for standardized input features.
    - Layout: neurons with correlated activation traces are positioned nearby.
    - White-ringed neurons: chosen trace through the strongest final incoming edge.

    This is a rendering of an *attribution*, not an intervention or causal path.
    All values refer to the output logit, relative to the specified baseline.
    Only 2 hidden layers are presently rendered; the explanation engine supports
    more layers.
    """
    if len(explanation.edge_contributions)!=3:
        raise ValueError('The oblique renderer currently supports exactly two hidden layers.')
    n_input,n_h1=explanation.edge_contributions[0].shape
    n_h2=explanation.edge_contributions[1].shape[1]
    if feature_names is None:
        feature_names=[f'x{i+1}' for i in range(n_input)]
    if len(feature_names)!=n_input:
        raise ValueError('feature_names must match n_features.')
    if n_input>18 or n_h1>36 or n_h2>25:
        raise ValueError('For compact 2.5D rendering, use <=18 inputs and <=36/25 hidden neurons.')
    sigs=activation_signatures(model,X_reference)
    if grid_layout is None:
        grids=[_layout_by_responses(sig) for sig in sigs]
    else:
        grids=grid_layout
    layers=explanation.node_contributions
    max_val=np.max(np.abs(np.concatenate([v.ravel() for v in layers[:-1]])))
    scale=max(float(max_val),.12)
    edge_scale=max(float(np.percentile(np.abs(np.concatenate([e.ravel() for e in explanation.edge_contributions])),96)),.05)
    if ax is None:
        fig,ax=plt.subplots(figsize=(7.2,6.6),dpi=200)
    else:
        fig=ax.figure
    fig.patch.set_facecolor('white');ax.set_facecolor('white')
    ax.set_aspect('equal');ax.set_xlim(-.045,1.075);ax.set_ylim(.055,1.017);ax.axis('off')
    fig.subplots_adjust(left=.012,right=.99,top=.992,bottom=.012)
    input_x=.15
    iny=np.linspace(.84,.20,n_input)
    anchors=[np.array([.350,.83]),np.array([.685,.745])]
    out=np.array([1.006,.45])
    centers=[];tiles=[]
    for ((nr,nc),layout),anchor in zip(grids,anchors):
        # Sized to each grid, so 16- and 9-neuron lattices retain same perspective.
        dv=np.array([.033,-.073])
        dh=np.array([.055,-.017])
        def pt(r,c): return anchor+r*dv+c*dh
        p={k:pt(*layout[k]) for k in layout}
        shape={k:np.array([pt(layout[k][0]-.43,layout[k][1]-.43),
                           pt(layout[k][0]-.43,layout[k][1]+.43),
                           pt(layout[k][0]+.43,layout[k][1]+.43),
                           pt(layout[k][0]+.43,layout[k][1]-.43)]) for k in layout}
        centers.append(p);tiles.append(shape)
    # All edges from the trained network are displayed; color is LOCAL attribution.
    group_positions=[{k:np.array([input_x,iny[k]]) for k in range(n_input)},*centers,{0:out}]
    lines=[];line_colors=[];lws=[]
    for l,contrib in enumerate(explanation.edge_contributions):
        for i in range(contrib.shape[0]):
            for j in range(contrib.shape[1]):
                amount=float(contrib[i,j]);v=min(abs(amount)/edge_scale,1.8)
                if not show_all_edges and v<.11:continue
                lines.append([group_positions[l][i],group_positions[l+1][j]])
                color=POS if amount>=0 else NEG
                line_colors.append((*to_rgb(color), .055+.25*min(v,1)))
                lws.append(.36+.57*min(v,1.4))
    ax.add_collection(LineCollection(lines,colors=line_colors,linewidths=lws,zorder=1))
    # Highlight a real path (not a unique causal explanation).
    path=[]
    if highlight:
        h2=int(np.argmax(np.abs(explanation.edge_contributions[-1][:,0])))
        h1=int(np.argmax(np.abs(explanation.edge_contributions[1][:,h2])))
        inp=int(np.argmax(np.abs(explanation.edge_contributions[0][:,h1])))
        path=[(0,inp,h1),(1,h1,h2),(2,h2,0)]
        for l,i,j in path:
            p0,p1=group_positions[l][i],group_positions[l+1][j]
            amount=explanation.edge_contributions[l][i,j]
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color='white',lw=4.2,alpha=.65,zorder=2)
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=POS if amount>=0 else NEG,lw=2.6,alpha=.95,zorder=3)
    # Thin shadow and plate for each hidden layer, visually distinct from model geometry.
    for l,((nr,nc),mapping) in enumerate(grids):
        shapes=tiles[l]
        footprint=np.concatenate(list(shapes.values()))
        # Draw the shadows below every tile; background is intentionally minimal.
        hull=np.array([footprint[np.argmin(footprint[:,0])],footprint[np.argmax(footprint[:,1])],
                       footprint[np.argmax(footprint[:,0])],footprint[np.argmin(footprint[:,1])]])
        ax.add_patch(Polygon(hull+np.array([.020,-.043]),closed=True,facecolor='#203F63',
                             linewidth=0,alpha=.045,zorder=4))
        for k in sorted(mapping, key=lambda i:sum(mapping[i])):
            points=shapes[k];p=centers[l][k];
            val=float(layers[l+1][k]);color=_signed_color(val,scale)
            low=points+np.array([.014,-.038])
            depth=7+sum(mapping[k])*.03
            ax.add_patch(Polygon([points[1],points[2],low[2],low[1]],closed=True,
                                 facecolor=_shade(color,.72),edgecolor='#28415F',linewidth=.45,zorder=depth))
            ax.add_patch(Polygon([points[2],points[3],low[3],low[2]],closed=True,
                                 facecolor=_shade(color,.84),edgecolor='#28415F',linewidth=.45,zorder=depth+.01))
            ax.add_patch(Polygon(points,closed=True,facecolor=color,edgecolor='#294261',
                                 linewidth=.55,zorder=depth+.02))
            ax.add_patch(Circle(p,.0044,facecolor=INK,edgecolor='none',zorder=depth+.05))
    # Rings drawn on top of tiles.
    if highlight:
        for layer,k in ((0,path[0][2]),(1,path[1][2])):
            p=centers[layer][k]
            ax.add_patch(Circle(p,.014,facecolor='white',edgecolor=INK,linewidth=1.15,zorder=35))
    for i,y in enumerate(iny):
        val=float(layers[0][i]);p=np.array([input_x,y]);c=_signed_color(val,scale)
        ax.add_patch(Circle(p,.016,facecolor=c,edgecolor=INK,linewidth=.65,zorder=30))
        ax.text(input_x-.027,y,str(feature_names[i]),fontsize=9.0,ha='right',va='center',color=INK)
    outcolor=_signed_color(explanation.logit_change,max(abs(explanation.logit_change),.1))
    ax.add_patch(Circle(out,.029,facecolor=outcolor,edgecolor=INK,lw=1.2,zorder=40))
    ax.text(out[0],out[1]+.051,'P(Y=1)',fontsize=9.3,color=INK,ha='center')
    ax.text(out[0],out[1]-.054,f'{explanation.probability:.3f}',fontsize=9.2,color=INK,ha='center')
    ax.text(.153,.955,f'{n_input} inputs',fontsize=9.5,color=INK,ha='center')
    ax.text(.45,.961,f'Hidden · {n_h1}',fontsize=9.5,color=INK,ha='center')
    ax.text(.83,.885,f'Hidden · {n_h2}',fontsize=9.5,color=INK,ha='center')
    ax.text(.96,.265,f'Δlogit  {explanation.logit_change:+.2f}',fontsize=8.0,ha='center',color=INK)
    # Minimal signed-meaning key, with no slide title, panel label, or boxed legend.
    ax.plot([.235,.254],[.108,.108],lw=3.3,color=NEG,solid_capstyle='round')
    ax.text(.265,.108,'decreases logit',fontsize=7.7,va='center',color='#586E81')
    ax.plot([.455,.474],[.108,.108],lw=3.3,color=POS,solid_capstyle='round')
    ax.text(.484,.108,'increases logit',fontsize=7.7,va='center',color='#586E81')
    ax.text(.53,.071,'edge / tile color: integrated contribution  ·  proximity: activation similarity',
            fontsize=7.6,color='#65788B',ha='center')
    return fig,ax
