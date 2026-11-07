"""Vector-native oblique 3D renderer for model information geometry."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .behavior_geometry import ProbabilityAtlas
from .functional_plot import _prism, _setup, INK


def plot_probability_atlas(atlas: ProbabilityAtlas, *, min_stability=.35,
                           figsize=(7.0,5.4), family=None, label_map=None):
    """Plot models by predictive Fisher-Rao similarity (or chosen metric).

    Prism positions are 3D metric-MDS approximations; screen-space distances
    are not equal to functional distances. Edges encode bootstrap-stable
    nearest neighbors under *resampling reference observations only*.
    Glyph shape is decorative; prism dimensions are fixed.
    """
    C=atlas.coordinates.copy(); n=len(C)
    rng=max(np.max(np.abs(C)),1e-9)
    C*=1.5/rng
    # Rigid camera rotation improves separation of genuinely distinct models.
    # The underlying 3D MDS coordinates/distances are unchanged.
    az,el=np.deg2rad((90,35))
    horizontal=np.array([np.cos(az),np.sin(az),0.])
    vertical=np.array([-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)])
    screen=np.c_[C@horizontal,C@vertical]
    base=np.c_[screen[:,0],-np.sin(az)*np.sin(el)*C[:,0]+np.cos(az)*np.sin(el)*C[:,1]]
    family=family or {}
    label_map=label_map or {}
    palette={'GLM':'#3F76B0','MLP':'#DBA24A','Forest':'#368E8C',
             'Kernel':'#826BB4','Bayes':'#A07587','Other':'#627F9D'}
    plt.rcParams['svg.fonttype']='none'
    fig,ax=plt.subplots(figsize=figsize,dpi=230)
    _setup(ax)
    for i in range(n):
        if abs(base[i,1]-screen[i,1])>.03:
            ax.plot([base[i,0],screen[i,0]],[base[i,1],screen[i,1]],color='#B4C0CB',lw=1.,alpha=.7,zorder=.5)
            ax.scatter([base[i,0]],[base[i,1]],s=7,facecolors='white',edgecolors='#9FAFBE',lw=.6,zorder=.7)
    # Stable nearest-neighbor links; darker lines mean stable under reweighting
    for i in range(n):
        for j in range(i+1,n):
            stability=atlas.edge_frequencies[i,j]
            if stability < min_stability: continue
            coords=screen[[i,j]]
            ax.plot(coords[:,0],coords[:,1]-.12,
                    color='#466B87',lw=.4+2.8*stability,alpha=.1+.7*stability,zorder=1.2,
                    solid_capstyle='round')
    for i in np.argsort(screen[:,1])[::-1]:
        x,y=screen[i]
        name=atlas.unique_names[i]
        fam=family.get(name,'Other')
        color=palette.get(fam,palette['Other'])
        _prism(ax,x,y,.155,.27,color,zorder=4+i*.005)
        group=atlas.equivalence_classes[i]
        label=label_map.get(name,name)
        if len(group)>1:
            label+=' ×'+str(len(group))
            ax.add_patch(Circle((x,y+.005),.127,fc='none',ec='white',lw=1.8,zorder=8))
            ax.add_patch(Circle((x,y+.005),.150,fc='none',ec=INK,lw=.9,zorder=8.1))
        ax.text(x,y-.006,label,ha='center',va='center',fontsize=7.5,
                fontweight='medium',color='white',zorder=9)
    x0,x1=screen[:,0].min(),screen[:,0].max()
    y0,y1=screen[:,1].min(),screen[:,1].max()
    ax.set_xlim(x0-.68,x1+.68)
    ax.set_ylim(y0-.77,y1+.68)
    metric_label={'fisher_rao':'Fisher–Rao', 'probability_rms':'probability RMS', 'hellinger':'Hellinger'}[atlas.geometry]
    ax.text(.018,.055,f'POSITION  ·  {metric_label} distance between predicted distributions',
            transform=ax.transAxes,color='#5A6F82',fontsize=8.5)
    ax.text(.018,.026,'LINKS  ·  nearest-neighbor stability under reference-sample resampling',
            transform=ax.transAxes,color='#5A6F82',fontsize=8.2)
    ax.text(.988,.026,f'3D stress  {atlas.distance_stress:.3f}',
            transform=ax.transAxes,color='#5A6F82',fontsize=8.2,ha='right')
    fig.subplots_adjust(.014,.018,.986,.978)
    return fig,ax
