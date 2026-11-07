"""Controlled, entirely synthetic photo-z representation comparison.

Run: python examples/adaptive_photoz_comparison.py
Outputs: docs/figures/adaptive_* (PNG, SVG, PDF) and docs/adaptive_metrics.json.
Training uses *photometry only*. True redshifts are held out for audit metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from modelatlas.som import train_som
from modelatlas.adaptive import growing_neural_gas, train_gtm, partition_tail_bounds
from modelatlas.adaptive_plot import plot_graph_atlas

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'/'figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                     'pdf.fonttype':42,'axes.linewidth':0.7,'savefig.transparent':False})


def generate_mock(n=6000, seed=39):
    """Toy 8-band SED photometry with redshift/type/dust degeneracies.

    These are analytic, band-averaged templates, not survey throughput curves.
    They do not model real HSC/LSST errors or realistic spectroscopic selection.
    """
    rng=np.random.default_rng(seed)
    source=rng.choice(3,size=n,p=[0.48,0.34,0.18])
    z=np.empty(n)
    z[source==0]=rng.normal(0.52,0.22,np.sum(source==0))
    z[source==1]=rng.normal(1.28,0.31,np.sum(source==1))
    z[source==2]=rng.normal(2.01,0.25,np.sum(source==2))
    z=np.clip(z,0.03,2.65)
    gal_type=rng.choice(3,size=n,p=[.40,.40,.20])
    dust=np.clip(rng.lognormal(-1.6,0.6,n)+(gal_type==1)*0.14,0,0.75)
    slope=rng.normal(0.2,0.25,n)+np.choose(gal_type,[-.55,.25,.65])
    br=np.clip(rng.normal(0.5,.20,n)+np.choose(gal_type,[.6,-.25,0.05]),0.02,1.8)
    # Band centres (um): u,g,r,i,z,y,J,H, approximate, not precise throughputs.
    lam=np.array([.37,.48,.62,.75,.88,.98,1.23,1.58])
    # Smooth response over 11 wavelengths per band.
    qs=np.linspace(-1.65,1.65,11)
    rest=lam[None,:,None]*(1.+qs[None,None,:]*.085)/(1.+z[:,None,None])
    continuum=(rest/.55)**slope[:,None,None]
    break_step=1.0 + br[:,None,None]/(1.+np.exp(-(rest-.4)/.018))
    atten=np.exp(-dust[:,None,None]*(np.maximum(.20,rest)/.55)**(-1.1))
    # Emission features: mild broad-band line enhancement.
    lines=np.array([.3727,.4861,.5007,.6563])
    strength=np.choose(gal_type,[.09,.30,.13]) + rng.uniform(0,.06,n)
    line_factor=1.+strength[:,None,None]*np.sum(np.exp(-.5*((rest[:,:,:,None]-lines[None,None,None,:])/.018)**2),axis=-1)
    flux=np.sum(continuum*break_step*atten*line_factor * np.exp(-.5*qs[None,None,:]**2),axis=2) / np.sum(np.exp(-.5*qs**2))
    intrinsic=rng.normal(0.,0.75,n)
    i_mag=20.4 + 2.1*np.log10(1.+z) + 1.15*z - intrinsic
    # Measured colors include heteroscedastic photometric noise.
    signal_to_noise=np.clip(42*np.exp(-.72*(i_mag-20.4)),4.2,70)
    flux=flux*(1.+rng.normal(0,0.85/signal_to_noise[:,None],flux.shape))
    flux=np.maximum(flux,0.007)
    mag=-2.5*np.log10(flux)
    colors_obs=mag[:,:-1]-mag[:,1:]
    X=np.column_stack([colors_obs, i_mag+rng.normal(0,.08,n)])
    # Target selection depends on magnitude, colors, and redshift-dependent
    # spectroscopic success; true z enters *the simulation only*.
    bright=1./(1.+np.exp(np.clip((i_mag-22.3)*1.15,-20,20)))
    color_priority=1./(1.+np.exp(np.clip(1.4*(colors_obs[:,2]-.35),-15,15)))
    p_attempt=np.clip(.055+.88*bright*color_priority,0.015,.95)
    attempted=rng.random(n)<p_attempt
    p_success=1./(1.+np.exp(np.clip(-2.5 + 0.58*(i_mag-21.5) + .83*(z-1.1)**2 + .55*(gal_type==1),-20,20)))
    usable=attempted&(rng.random(n)<p_success)
    z_spec=np.where(usable,z,np.nan)
    return {'X':X,'z':z,'z_spec':z_spec,'attempted':attempted,'usable':usable,
            'gal_type':gal_type,'z0':1.2,'photometric_bands':lam}


def evaluate(name, prototypes, assignments, z_true, z_spec, x_test, adjacency, gamma=0.16):
    n_nodes=len(prototypes)
    dist=pairwise_distances(x_test,prototypes)
    near=np.argpartition(dist,1,axis=1)[:,:2]
    best=np.argmin(dist,axis=1)
    a=best
    b=np.where(near[:,0]==a,near[:,1],near[:,0])
    qe=float(np.mean(dist[np.arange(len(x_test)),a]))
    topo=float(np.mean(~adjacency[a,b]))
    bounds=partition_tail_bounds(assignments,z_spec,n_nodes,z0=1.2,gamma=gamma)
    pop=np.bincount(assignments,minlength=n_nodes)
    truth_q=np.mean(z_true>1.2)
    within_std=np.sum([len(z_true[assignments==k])*np.std(z_true[assignments==k])
                      for k in range(n_nodes) if np.any(assignments==k)])/len(z_true)
    # Audit of W1 assumption using hidden true redshifts (NEVER in bounds):
    # Distributions with usable and missing spectra within populated nodes.
    viol=0; comparable=0; median_w1=[]
    for k in np.flatnonzero(pop):
        successful=(assignments==k)&np.isfinite(z_spec)
        missing=(assignments==k)&~np.isfinite(z_spec)
        if successful.sum()>1 and missing.sum()>1:
            d=wasserstein_distance(z_true[successful],z_true[missing])
            median_w1.append(d)
            comparable+=pop[k]
            if d>gamma: viol+=pop[k]
    return {'name':name,'nodes':n_nodes,'edges':int(np.triu(adjacency).sum()),
            'test_quantization_error':qe,'test_topographic_error':topo,
            'weighted_within_cell_z_sd':float(within_std),
            'occupied_nodes':int(np.sum(pop>0)),'nodes_without_spectra':bounds['n_empty_spec'],
            'identification_interval':[bounds['lower'],bounds['upper']],
            'identification_width':bounds['width'],
            'true_highz_fraction':float(truth_q),
            'interval_contains_truth':bool(bounds['lower']<=truth_q<=bounds['upper']),
            'audit_fraction_W1_violation_among_comparable_nodes':float(viol/comparable) if comparable else None,
            'audit_comparable_population_fraction':float(comparable/len(z_true)),
            'median_true_W1_between_spec_and_missing':float(np.median(median_w1)) if median_w1 else None}


def adjacency_grid(shape):
    rows,cols=shape
    n=rows*cols
    A=np.zeros((n,n),bool)
    for r in range(rows):
        for c in range(cols):
            i=r*cols+c
            for rr,cc in [(r+1,c),(r,c+1)]:
                if rr<rows and cc<cols:
                    j=rr*cols+cc;A[i,j]=A[j,i]=True
    return A


def shade(c, k=1.):
    arr=np.asarray(colors.to_rgb(c))
    return np.clip(arr*k,0,1)


def cube(center, size=(.11,.11,.11), color='#2B78A6', alpha=1.,edge='#25415A'):
    x,y,z=center; sx,sy,sz=size
    X=np.array([[x-sx,y-sy,z-sz],[x+sx,y-sy,z-sz],[x+sx,y+sy,z-sz],[x-sx,y+sy,z-sz],
                [x-sx,y-sy,z+sz],[x+sx,y-sy,z+sz],[x+sx,y+sy,z+sz],[x-sx,y+sy,z+sz]])
    faces=[X[[4,5,6,7]],X[[0,1,5,4]],X[[1,2,6,5]],X[[2,3,7,6]],X[[3,0,4,7]]]
    fcs=[shade(color,k) for k in [1.13,.78,.68,.75,.68]]
    return Poly3DCollection(faces,facecolors=fcs,edgecolors=edge,linewidths=.55,alpha=alpha,zsort='average')


def plot_graph(graph, X, assignments, obs_z, output, seed=39):
    """Vector-native oblique graph: 3D positions from PCA of 8D prototypes.

    3D coordinates are a *projection*, edges are GNG learned adjacency.
    Node color is spectroscopic coverage, and size is target occupancy.
    Missing spectra are not treated as zero redshift. Input connections shown
    for selected prototypes are literal codebook feature components.
    """
    w=graph.prototypes
    pca=PCA(n_components=3,svd_solver='full')
    xyz=pca.fit_transform(w)
    xyz=xyz/(np.std(xyz,axis=0).clip(.3))*np.array([1.4,.92,.68])
    n=len(w)
    pop=np.bincount(assignments,minlength=n)
    spec=np.bincount(assignments[np.isfinite(obs_z)],minlength=n)
    cov=np.divide(spec,pop,out=np.zeros(n,float),where=pop>0)
    fig=plt.figure(figsize=(7.0,6.4),facecolor='white')
    ax=fig.add_axes([.035,.04,.92,.92],projection='3d',computed_zorder=False)
    ax.set_facecolor('white')
    ax.view_init(elev=21,azim=-61,roll=0)
    ax.set_proj_type('ortho')
    cmap=plt.get_cmap('viridis')
    # Learned graph connections lie in the projected 3D PCA space.
    for a,b in graph.edges:
        aa,bb=xyz[a],xyz[b]
        ax.plot([aa[0],bb[0]],[aa[1],bb[1]],[aa[2],bb[2]],
                color='#73869A',alpha=.34,lw=1.0,zorder=2)
    # Nodes are small extruded cuboids; size indicates target occupancy.
    for k,coord in enumerate(xyz):
        if pop[k]==0:color='#D7DDE2'
        else:color=cmap(.12+.8*cov[k])
        rad=.095+.075*np.sqrt(pop[k]/max(pop.max(),1))
        ax.add_collection3d(cube(coord,(rad,rad,rad*.58),color=color,edge='#25374A'))
    ax.set(xlim=(xyz[:,0].min()-.5,xyz[:,0].max()+.6),
           ylim=(xyz[:,1].min()-.5,xyz[:,1].max()+.5),
           zlim=(xyz[:,2].min()-.5,xyz[:,2].max()+.5))
    ax.set_box_aspect((1.36,1.,.82),zoom=.92)
    ax.set_axis_off()
    # Tiny annotation outside data surface, no full-slide elements.
    fig.text(.055,.92,'Growing neural gas',fontsize=13,fontweight='medium',color='#1A2C3D')
    fig.text(.055,.875,'learned edges · 8D prototypes',fontsize=9,color='#526477')
    cax=fig.add_axes([.32,.055,.36,.014])
    cb=plt.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,orientation='horizontal')
    cb.set_label('Usable spectroscopy / target objects',fontsize=8,color='#526477',labelpad=5)
    cb.ax.tick_params(labelsize=7,length=2)
    for ext in ['png','svg','pdf']:
        fig.savefig(output.with_suffix('.'+ext),dpi=250,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)
    return pca.explained_variance_ratio_.sum()


def plot_comparison(som_w,gtm_w,graph, X, assignments, zspec, output):
    """One PNG/PDF comparison; each plot remains an independent small figure."""
    weights=[som_w,gtm_w,graph.prototypes]
    labels=['SOM · fixed lattice','GTM · probabilistic','GNG · adaptive graph']
    adj=[adjacency_grid((8,8)),adjacency_grid((8,8)),graph.adjacency]
    # Shared 2D PCA fit for all prototypes, avoids different coordinate systems.
    pc=PCA(n_components=2).fit(X)
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.5),constrained_layout=True)
    cmap=plt.get_cmap('viridis')
    for ax,w,a,assignment,label in zip(axes,weights,adj,assignments,labels):
        xy=pc.transform(w)
        n=np.bincount(assignment,minlength=len(w))
        ns=np.bincount(assignment[np.isfinite(zspec)],minlength=len(w))
        fraction=np.divide(ns,n,out=np.zeros(len(w),float),where=n>0)
        for u,v in np.column_stack(np.where(np.triu(a,1))):
            ax.plot([xy[u,0],xy[v,0]],[xy[u,1],xy[v,1]],c='#9CAEC0',lw=.5,alpha=.48,zorder=1)
        ax.scatter(xy[:,0],xy[:,1],s=15+48*np.sqrt(n/max(n.max(),1)),
                   c=fraction,cmap=cmap,vmin=0,vmax=1,zorder=3,ec='#254159',lw=.38)
        ax.set_title(label,fontsize=10,fontweight='medium',loc='left',pad=8)
        ax.set_xlabel('Photometric PC1',fontsize=8)
        if ax is axes[0]:ax.set_ylabel('Photometric PC2',fontsize=8)
        ax.tick_params(labelsize=7,length=2)
        ax.set_aspect('equal',adjustable='datalim')
        for spine in ('top','right'):ax.spines[spine].set_visible(False)
    for ext in ['png','svg','pdf']:
        fig.savefig(output.with_suffix('.'+ext),dpi=250,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)


def main(seed=39):
    d=generate_mock(6000,seed)
    X=d['X'];z=d['z'];sz=d['z_spec']
    train_index,test_index=train_test_split(np.arange(len(z)),test_size=.25,random_state=seed)
    scaler=StandardScaler().fit(X[train_index]); x=scaler.transform(X)
    # Unsupervised fit: ALL prototype methods see same training photometry only.
    som=train_som(x[train_index],grid=(8,8),steps=7500,seed=seed)
    gtm=train_gtm(x[train_index],grid=(8,8),n_iter=45,ridge=.005,seed=seed)
    graph=growing_neural_gas(x[train_index],max_nodes=64,steps=15500,
                            insertion_interval=230,max_edge_age=10,seed=seed)
    somw=som.reshape(-1,x.shape[1]);gtmw=gtm.prototypes; gngw=graph.prototypes
    wgts=[somw,gtmw,gngw]
    ass=[np.argmin(pairwise_distances(x,w),axis=1) for w in wgts]
    aadj=[adjacency_grid((8,8)),adjacency_grid((8,8)),graph.adjacency]
    res=[evaluate(name,w,ids,z,sz,x[test_index],a,gamma=.16)
         for name,w,ids,a in zip(['SOM','GTM','Growing Neural Gas'],wgts,ass,aadj)]
    fraction=float(np.mean(z>1.2))
    naive=float(np.mean(sz[np.isfinite(sz)]>1.2))
    summary={'seed':seed,'n':len(z),'n_training_photometry':len(train_index),
             'n_test_photometry':len(test_index),'n_attempted':int(d['attempted'].sum()),
             'n_usable_spectra':int(np.isfinite(sz).sum()),'z_threshold':1.2,
             'gamma_W1_redshift_units':.16,'full_population_true_tail_fraction':fraction,
             'successful_spec_tail_fraction':naive,
             'comparisons':res,'GTM_regularized_beta':gtm.beta}
    (ROOT/'docs').mkdir(exist_ok=True)
    (ROOT/'docs'/'adaptive_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    explained=plot_graph(graph,x,ass[2],sz,OUT/'adaptive_gng_3d',seed)
    _,_,meta=plot_graph_atlas(graph,ass[2],sz,feature_labels=['u-g','g-r','r-i','i-z','z-y','y-J','J-H','i mag'],output=OUT/'adaptive_graph_architecture')
    plt.close('all')
    summary['atlas_rendering']=meta
    summary['PCA_3D_variance_explained_graph_prototypes']=float(explained)
    (ROOT/'docs'/'adaptive_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_comparison(somw,gtmw,graph,x,ass,sz,OUT/'adaptive_three_models')
    # Tiny companion diagnostic: n(z) truth (audit) vs usable spectra only.
    fig,ax=plt.subplots(figsize=(4.0,2.65))
    bins=np.linspace(0,2.65,42)
    ax.hist(z,bins=bins,density=True,histtype='step',lw=1.8,color='#27477A',label='Target (truth audit)')
    ax.hist(sz[np.isfinite(sz)],bins=bins,density=True,histtype='step',lw=1.6,color='#D36B40',label='Usable spectra')
    ax.axvline(1.2,color='#68717C',lw=.8,ls='--')
    ax.set_xlabel('Redshift');ax.set_ylabel('Density')
    ax.legend(frameon=False,fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    for ext in ['png','svg','pdf']:
        fig.savefig((OUT/'adaptive_selection').with_suffix('.'+ext),dpi=250,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()