"""Compact vector-native Matplotlib visual grammar for functional geometry."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from .functional import FunctionalMap, InformationLattice

INK = '#273449'
BLUE = '#3E749C'
PALE = '#91A2B3'


def _mix(color, target, fraction):
    from matplotlib.colors import to_rgb
    a, b = np.array(to_rgb(color)), np.array(to_rgb(target))
    return tuple((1-fraction)*a+fraction*b)


def _corners(x, y, radius, angle=np.pi/6):
    t = angle + np.arange(6) * np.pi/3
    return np.column_stack((x+radius*np.cos(t), y+radius*.90*np.sin(t)))


def _prism(ax, x, y, radius, height, face, *, zorder=3, edge='#FFFFFF', alpha=1., line_width=.85):
    """Hexagonal prism built from ordinary 2D vector polygon artists."""
    top = _corners(x,y,radius)
    bottom = top - np.array((.07*height, .68*height))
    # light background shadow
    ax.add_patch(Polygon(bottom+np.array((.075,-.04)), closed=True, color='#1D293C',alpha=.095,zorder=zorder-.5,lw=0))
    for i in range(6):
        j=(i+1)%6
        if (top[i,0]+top[j,0])/2 < x-0.015:
            fill=_mix(face, '#26364F', .38)
        else:
            fill=_mix(face, '#17243A', .17)
        polygon=np.vstack([top[i],top[j],bottom[j],bottom[i]])
        ax.add_patch(Polygon(polygon, facecolor=fill,edgecolor=_mix(fill,'#2A3447',.20),lw=.3,zorder=zorder+.02,alpha=alpha))
    ax.add_patch(Polygon(top,closed=True,facecolor=face,edgecolor=edge,lw=line_width,zorder=zorder+.5,alpha=alpha,joinstyle='round'))
    ax.add_patch(Circle((x-.14*radius,y+.13*radius),radius*.09,fc='#FFFFFF',ec='none',alpha=.55,zorder=zorder+.7))
    return top


def _setup(ax):
    ax.set_axis_off()
    ax.set_facecolor('white')
    ax.set_aspect('equal',adjustable='box')
    return ax


def plot_functional_atlas(atlas: FunctionalMap, *, family=None, quality=None, selected=None,
                          figsize=(7.6, 5.8), seed=0):
    """Render an empirical functional model quotient via oblique hex prisms.

    Base coordinates are 3D classical metric MDS of prediction vectors.
    The oblique camera is display-only: pixel distances are *not* exact
    functional distances. Report 3D stress separately from the screenshot.
    Hexagons are glyphs, not learned SOM or model topology. Edges are a 2-NN
    graph of fitted prediction profiles, NOT actual model connections.
    """
    names=atlas.names
    n=len(names)
    if n<2:raise ValueError('at least two model predictions required')
    # Three principal coordinates of actual held-out prediction profiles.
    # The isometric projection is *display-only* and has its own distortions.
    xyz=atlas.coords3.copy()
    scale=1.70/max(float(np.max(np.abs(xyz))),1e-9)
    xyz*=scale
    screen=np.c_[1.28*xyz[:,0]+.42*xyz[:,1],
                 -.18*xyz[:,0]+.62*xyz[:,1]+1.25*xyz[:,2]]
    ground=np.c_[1.28*xyz[:,0]+.42*xyz[:,1],
                 -.18*xyz[:,0]+.62*xyz[:,1]]
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(figsize=figsize,dpi=220)
    _setup(ax)
    family=family or {name:'other' for name in names}
    cmap={'GLM':'#2B6698','MLP':'#D6A052','Forest':'#34958D','Kernel':'#8F78B3','Bayes':'#9B7287','Other':'#65819F'}
    # Model glyphs all have identical geometric depth. Coordinates encode
    # functional behavior, not model performance or uncertainty.
    quality=quality or {}
    # Vertical guide stems show the third functional coordinate relative to
    # the plane of the first two functional coordinates.
    for i in range(n):
        if abs(screen[i,1]-ground[i,1]) > .035:
            ax.plot((screen[i,0],ground[i,0]),(screen[i,1],ground[i,1]),
                    color='#94A6BA',lw=1.0,alpha=.47,zorder=.52)
            ax.scatter([ground[i,0]],[ground[i,1]],s=8,facecolors='white',
                       edgecolors='#8E9CAC',lw=.6,zorder=.54)
    # Pairwise k-nearest links mean predictive similarity, not network weights
    edges=set()
    for i in range(n):
        neighbors=np.argsort(atlas.distances[i]+np.eye(n)[i]*1000)[:2] if n>3 else [j for j in range(n) if j!=i]
        for j in neighbors:
            if atlas.distances[i,j] >= 1e-10:
                edges.add(tuple(sorted((i, int(j)))))
    for i,j in sorted(edges):
        ax.plot(screen[[i,j],0],screen[[i,j],1]-.16,
                color='#8DA4BA',lw=1.3,alpha=.30,zorder=.8)
    # ellipse / orbit halo for empirical equal predictions; do not imply proof of global equivalence
    coincident=[]
    for i in range(n):
        dup=[j for j in range(n) if atlas.distances[i,j]<1e-10]
        if len(dup)>1 and i==dup[0]:coincident.append((i,dup))
    active=set()
    for idx,dup in coincident:
        active.update(dup[1:])
    # scale node areas as response disagreement (meaningful, though illustrative)
    for i in sorted((j for j in range(n) if j not in active),key=lambda j:screen[j,1],reverse=True):
        x,y=screen[i]
        r=.169
        h=.27
        color=cmap.get(family.get(names[i],'Other'),'#65819F')
        _prism(ax,x,y,r,h,color,zorder=4+.001*i)
        if any(z==i for z,_ in coincident):
            ax.add_patch(Circle((x,y+.03),r*.69,fc='none',ec='#FFFFFF',lw=1.9,zorder=7))
            ax.add_patch(Circle((x,y+.03),r*.87,fc='none',ec=INK,lw=1.0,zorder=7.2))
        # Short model-family tags keep the graph readable even when model
        # predictions are close. Full names / held-out diagnostics in the JSON.
        tags={'GLM':'GLM','Regularized GLM':'rGLM','MLP A':'MLP',
              'MLP B':'MLPb','Random forest':'RF','Extra Trees':'ET',
              'RBF SVM':'SVM','KNN':'kNN','Gaussian NB':'NB',
              'Ridge':'Ridge','Random forest R':'RF','SVR':'SVR',
              'kNN R':'kNN','Boosted R':'Boost','Tree R':'Tree','Mean':'Mean'}
        label=tags.get(names[i], names[i][:6])
        if any(z==i for z,_ in coincident): label='MLP ×2'
        ax.text(x,y-.010,label,ha='center',va='center',fontsize=6.7,
                color='#FFFFFF' if family.get(names[i])!='GLM' else '#FFFFFF',
                fontweight='medium',zorder=10)

    xmin,xmax=np.min(screen[:,0]),np.max(screen[:,0]);ymin,ymax=np.min(screen[:,1]),np.max(screen[:,1]);
    ax.set_xlim(xmin-.80,xmax+.95)
    ax.set_ylim(ymin-.90,ymax+.55)
    # Plot key, restrained scientific annotations
    ax.text(.02,.062,'3D POSITION  ·  held-out prediction similarity',transform=ax.transAxes,fontsize=7.9,color='#566678')
    ax.text(.02,.030,'PRISMS  ·  model families (not latent neurons)',transform=ax.transAxes,fontsize=7.9,color='#566678')
    ax.text(.99,.030,f'3D stress {atlas.stress3:.3f}',transform=ax.transAxes,fontsize=7.6,color='#647486',ha='right')
    fig.subplots_adjust(.018,.035,.982,.985)
    return fig,ax


def plot_information_lattice(lattice: InformationLattice, *, figsize=(9.2,5.7)):
    """Draw the Hasse diagram of observed finite-feature information partitions.

    Each hex is a subset of 4 measured features; the Hasse arrows are *true*
    set-inclusion cover relations, rather than learned graph/model weights.
    Prism height and color both encode average unresolved predictive variance.
    """
    m=len(lattice.feature_names)
    if m!=4:raise ValueError('compact example renderer currently supports four features')
    nodes=list(lattice.nodes)
    R={node.subset:node.residual_variance for node in nodes}
    base=max(lattice.baseline_variance,1e-12)
    position={}
    for k in range(m+1):
        this=[node.subset for node in nodes if len(node.subset)==k]
        # Use a symmetric Hasse-diagram layout with stable ordering
        for i,S in enumerate(this):
            y=(i-(len(this)-1)/2)*(1.11 if k==2 else 1.25)
            x=1.78*k
            # slightly shear lattice into oblique perspective
            position[S]=(x+.18*y,y*.92+.075*x)
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(figsize=figsize,dpi=220)
    _setup(ax)
    # Edges behind vertices. Line width carries average reduction in hidden variance.
    for S in position:
        for k in range(m):
            if k in S:continue
            T=tuple(sorted(S+(k,)))
            x1,y1=position[S];x2,y2=position[T]
            gain=max(0,R[S]-R[T])/base
            ax.plot((x1,x2),(y1-.25,y2-.25),color='#8498AF',lw=.78+3.3*gain,alpha=.20+.55*min(1,2*gain),zorder=.3)
    cmap=plt.get_cmap('cividis')
    for node in sorted(nodes,key=lambda q:position[q.subset][1],reverse=True):
        S=node.subset
        x,y=position[S]
        frac=np.clip(node.residual_variance/base,0,1)
        # dark blue = low hidden variance; golden = high hidden variance
        face=cmap(.12+.84*frac)
        h=.14+.59*np.sqrt(frac)
        _prism(ax,x,y,.255,h,face,zorder=4)
        label='∅' if not S else ''.join(lattice.feature_names[i] for i in S)
        ax.text(x,y+.33,label,color=INK,fontsize=8.5,ha='center',va='bottom',fontweight='medium',zorder=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.77,pad=.5))
    ax.set_xlim(-.62,8.1)
    ax.set_ylim(-3.95,4.0)
    ax.text(.015,.03,'MORE FEATURES  →  FINER PARTITION',transform=ax.transAxes,fontsize=8.8,color='#566678')
    ax.text(.99,.032,'Prism height + color = unresolved prediction variance',transform=ax.transAxes,fontsize=8.,color='#566678',ha='right')
    fig.subplots_adjust(.016,.028,.984,.982)
    return fig,ax


def plot_prediction_fibers(X:np.ndarray, prediction:np.ndarray, *, features=(0,1), n_bins=4,
                           labels=('input 1','input 2'), figsize=(6.35,6.1)):
    """A glyph for conditional prediction distributions at a fixed fitted model.

    This shows empirical quantile bands within *projected feature bins*, not
    uncertainty about an estimated model and not an identified set. Bins are
    based on quantiles in the supplied reference distribution.
    """
    X=np.asarray(X,float);p=np.asarray(prediction,float)
    if X.ndim!=2 or p.shape!=(len(X),) or len(features)!=2:raise ValueError('X and prediction dimensions mismatch')
    q=np.linspace(0,1,n_bins+1)
    cuts=[np.quantile(X[:,k],q) for k in features]
    bins=[np.searchsorted(cuts[j][1:-1], X[:,features[j]],side='right') for j in range(2)]
    arr=np.stack(bins,axis=1)
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(figsize=figsize,dpi=220);_setup(ax)
    stats=[]
    for i in range(n_bins):
        for j in range(n_bins):
            p0=p[(arr[:,0]==i)&(arr[:,1]==j)]
            if len(p0)<4:continue
            lo,md,hi=np.quantile(p0,[.1,.5,.9]);stats.append((i,j,lo,md,hi,len(p0)))
    for i,j,lo,md,hi,n in sorted(stats,key=lambda z:z[0]+z[1],reverse=True):
        x=(i-(n_bins-1)/2)*.83+(j-(n_bins-1)/2)*(-.43)
        y=(i-(n_bins-1)/2)*.28+(j-(n_bins-1)/2)*.64
        width=hi-lo
        col=plt.get_cmap('cividis')(.12+.8*min(1,width))
        _prism(ax,x,y,.31,.21,col,zorder=2)
        # vertical fibers of conditional prediction distribution (not model uncertainty)
        ybase=y+.075
        scale=1.95
        yl=ybase+scale*lo;yh=ybase+scale*hi;ym=ybase+scale*md
        xs=np.array([x-.095,x+.095])
        ax.add_patch(Polygon([(xs[0],yl),(xs[1],yl),(xs[1],yh),(xs[0],yh)],facecolor='#3D769D',edgecolor='none',alpha=.15,zorder=6))
        ax.plot([x,x],[yl,yh],color='#2B6588',lw=2.2,alpha=.74,zorder=7,solid_capstyle='round')
        ax.plot([x-.11,x+.11],[yl,yl],color='#2B6588',lw=1.,alpha=.74,zorder=7)
        ax.plot([x-.11,x+.11],[yh,yh],color='#2B6588',lw=1.,alpha=.74,zorder=7)
        ax.scatter([x],[ym],s=24,facecolor='#FAF9F4',edgecolor=INK,lw=.7,zorder=8)
    ax.text(.02,.04,'TILES  ·  bins in two displayed features',transform=ax.transAxes,fontsize=8,color='#526479')
    ax.text(.02,.008,'VERTICAL FIBERS  ·  10–90% of fixed-model predictions',transform=ax.transAxes,fontsize=8,color='#526479')
    ax.set_xlim(-2.8,2.8);ax.set_ylim(-2.2,4.10)
    fig.subplots_adjust(.01,.025,.99,.98)
    return fig,ax
