"""Matplotlib oblique scientific renderers for symmetry-aligned tanh MLPs.

A 2.5D deterministic rendering: neuron coordinates are shared comparison
coordinates, not spatial positions learned by a neural network. Shape/color/
connections are computed from fitted activations and weight matrices.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb, Normalize
from matplotlib.patches import Polygon, Circle

from .symmetry import activations

INK='#18324A'; SUB='#597082'; RUST='#BD6549'; BLUE='#397DA7'; WHITE='#FFFFFF'


def _shade(color,scale):
    return tuple(np.clip(np.array(to_rgb(color))*scale,0,1))


def _xy(org,row,col):
    return np.asarray(org) + row*np.array([.16,-.45]) + col*np.array([.49,-.135])


def _tile(ax,org,row,col,color,depth=.13,level=4):
    half=.46
    pts=np.array([_xy(org,row-half,col-half),_xy(org,row-half,col+half),
                  _xy(org,row+half,col+half),_xy(org,row+half,col-half)])
    side=np.array([.105,-depth])
    ax.add_patch(Polygon([pts[1],pts[2],pts[2]+side,pts[1]+side],facecolor=_shade(color,.68),edgecolor=INK,linewidth=.36,zorder=level))
    ax.add_patch(Polygon([pts[2],pts[3],pts[3]+side,pts[2]+side],facecolor=_shade(color,.8),edgecolor=INK,linewidth=.38,zorder=level+.02))
    ax.add_patch(Polygon(pts,facecolor=color,edgecolor='#19364c',linewidth=.58,zorder=level+.08))


def _settings(ax):
    ax.set_aspect('equal',adjustable='box');ax.axis('off');ax.set_facecolor('white')


def _network(ax,coefs,biases,x,base,tag,summary,highlight=True):
    layers,logits,prob=activations(coefs,biases,x[None,:])
    h1=layers[1].ravel();h2=layers[2].ravel()
    o1=np.array([2.80,base+2.25]);o2=np.array([5.6,base+2.00]);out=np.array([8.55,base+.92]);xin=.72
    ys=np.linspace(base+2.22,base-.25,8)
    n1=[_xy(o1,*divmod(i,4)) for i in range(16)];n2=[_xy(o2,*divmod(i,3)) for i in range(9)]
    lines=[];colors=[];widths=[]
    for W,pp,qq in [(coefs[0],[(xin,float(v)) for v in ys],n1),
                    (coefs[1],n1,n2),(coefs[2],n2,[out])]:
        norm=max(float(np.quantile(abs(W),.91)),1e-9)
        for i,u in enumerate(pp):
            for j,v in enumerate(qq):
                w=float(W[i,j]);value=min(abs(w)/norm,1.25)
                lines.append([u,v]);colors.append((*to_rgb(BLUE if w>0 else RUST), .043+.20*value))
                widths.append(.25+.41*value)
    ax.add_collection(LineCollection(lines,colors=colors,linewidths=widths,zorder=1))
    for i,yy in enumerate(ys):
        col=plt.get_cmap('coolwarm')(.5+.44*np.tanh(x[i]/2.0))
        ax.add_patch(Circle((xin,yy),.085,facecolor=col,edgecolor=INK,linewidth=.8,zorder=30))
    # Lattice positions are identical for both models, after symmetry matching.
    for org,values,width,nodes in [(o1,h1,4,n1),(o2,h2,3,n2)]:
        ax.add_patch(Polygon(np.asarray([_xy(org,-.52,-.52),_xy(org,-.52,width-.48),
                           _xy(org,len(nodes)//width-.48,width-.48),_xy(org,len(nodes)//width-.48,-.52)])+np.array([.07,-.23]),
                           facecolor='#4d6c82',edgecolor='none',alpha=.085,zorder=2))
        for k in sorted(range(len(values)),key=lambda k:sum(divmod(k,width))):
            r,c=divmod(k,width)
            color=plt.get_cmap('coolwarm')(.5+.45*np.clip(values[k],-1,1))
            z=4+(.14*r+.07*c)
            _tile(ax,org,r,c,color,level=z)
            p=nodes[k]
            ax.add_patch(Circle(p,.027,facecolor=INK,edgecolor='none',zorder=z+.20))
    if highlight:
        # This ranks output-layer logit terms h2_j * W_out,j (not causal attribution).
        scores=np.abs(h2*coefs[-1].ravel());inds=np.argsort(scores)[-3:]
        for rank,i in enumerate(inds):
            ax.add_patch(Circle(n2[i],.090,facecolor='white',edgecolor='#172F49',linewidth=1.35,zorder=23))
            ax.add_patch(Circle(n2[i],.032,facecolor=RUST if h2[i]*coefs[-1][i,0]>0 else BLUE,
                                edgecolor='none',zorder=24))
    col=plt.get_cmap('coolwarm')(.5+.42*(2*prob[0]-1))
    ax.add_patch(Circle(out,.150,facecolor=col,edgecolor=INK,linewidth=1.15,zorder=25))
    ax.text(out[0],out[1]-.25,f'{prob[0]:.3f}',ha='center',va='top',fontsize=11.5,color=INK)
    ax.text(.65,base+2.78,tag,fontsize=13,color=INK,weight='medium')
    ax.text(out[0],out[1]+.27,'P(Y=1)',ha='center',fontsize=9,color=SUB)
    ax.text(.65,base-0.43,summary,fontsize=9,color=SUB)


def _export(fig,outstem):
    outstem=Path(outstem);outstem.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg','pdf'):
        fig.savefig(outstem.with_suffix('.'+extension),dpi=250,facecolor='white',bbox_inches='tight',pad_inches=.14)
    plt.close(fig)


def plot_independent_pair(modelA,alignedB,x,title,label,metrics,directory):
    """Same input and layer coordinates; differences come from different fits."""
    fig,ax=plt.subplots(figsize=(9.6,10.25),dpi=200)
    fig.patch.set_facecolor('white');_settings(ax)
    ax.set_xlim(.24,9.2);ax.set_ylim(.05,9.85)
    ax.text(.65,9.68,f'{title}  ·  independent fits',fontsize=13.5,color=INK,weight='medium')
    s1=f"test AUC {metrics['test_auc_A']:.3f}  ·  seed {metrics['seed_A']}"
    s2=f"test AUC {metrics['test_auc_B']:.3f}  ·  seed {metrics['seed_B']}"
    _network(ax,modelA.coefs_,modelA.intercepts_,x,5.93,'A  ·  reference network',s1)
    _network(ax,alignedB.coefs,alignedB.intercepts,x,1.55,'B  ·  symmetry-aligned independent fit',s2)
    ax.plot([.57,8.87],[5.18,5.18],color='#e3eaf0',linewidth=1)
    diff=abs(metrics['selected_p_A']-metrics['selected_p_B'])
    ax.text(4.71,.38,f"Same observation; held-out prediction difference = {diff:.3f}",ha='center',fontsize=9.3,color=INK)
    ax.text(4.71,.10,'Tile color: tanh activation   ·   edge hue: signed weight   ·   dark rings: strongest output logit terms',ha='center',fontsize=7.6,color=SUB)
    _export(fig,Path(directory)/'independent_networks_aligned')


def plot_residual_atlas(modelA,alignedB,X_holdout,title,metrics,directory):
    """A single oblique atlas of residual hidden activation differences.

    RMS over held-out observations is displayed on the shared neuron grid.
    Edge differences are between *aligned parameters*, not attributions.
    """
    first=activations(modelA.coefs_,modelA.intercepts_,X_holdout)
    second=activations(alignedB.coefs,alignedB.intercepts,X_holdout)
    m1,m2=[np.sqrt(np.mean((first[0][i]-second[0][i])**2,axis=0)) for i in (1,2)]
    coefsA=modelA.coefs_;coefsB=alignedB.coefs
    fig,ax=plt.subplots(figsize=(9.0,5.6),dpi=200)
    fig.patch.set_facecolor('white');_settings(ax)
    ax.set_xlim(.15,9.25);ax.set_ylim(.05,4.50)
    o1=np.array([2.78,3.2]);o2=np.array([5.60,2.93]);out=np.array([8.5,1.64]);xin=.75
    ys=np.linspace(3.15,.65,8)
    n1=[_xy(o1,*divmod(i,4)) for i in range(16)]
    n2=[_xy(o2,*divmod(i,3)) for i in range(9)]
    lw=[];cols=[];lines=[]
    for A,B,a,b in [(coefsA[0],coefsB[0],[(xin,float(v)) for v in ys],n1),
                    (coefsA[1],coefsB[1],n1,n2),(coefsA[2],coefsB[2],n2,[out])]:
        d=B-A
        scale=max(float(np.quantile(abs(d),.95)),1e-7)
        for i,v in enumerate(a):
            for j,w in enumerate(b):
                delta=d[i,j];s=min(abs(delta)/scale,1.25)
                lines.append([v,w]);cols.append((*to_rgb(RUST if delta>0 else BLUE),.035+.23*s))
                lw.append(.25+.46*s)
    ax.add_collection(LineCollection(lines,colors=cols,linewidths=lw,zorder=1))
    for yy in ys:
        ax.add_patch(Circle((xin,yy),.082,facecolor='#e3e8eb',edgecolor=INK,lw=.7,zorder=25))
    vmax=max(float(np.quantile(np.r_[m1,m2],.98)),.03)
    for org,vals,width,points in [(o1,m1,4,n1),(o2,m2,3,n2)]:
        for k in sorted(range(vals.size),key=lambda k:sum(divmod(k,width))):
            r,c=divmod(k,width)
            color=plt.get_cmap('magma')(.17+.77*min(vals[k]/vmax,1))
            z=4+.12*r+.07*c
            _tile(ax,org,r,c,color,level=z)
            ax.add_patch(Circle(points[k],.025,facecolor='white',edgecolor='none',zorder=24))
    mean_pred=metrics['aligned_metrics_test']['mean_abs_prediction_difference']
    ax.add_patch(Circle(out,.16,facecolor='#f3ad62',edgecolor=INK,lw=1.2,zorder=25))
    ax.text(out[0],out[1]-.28,f'{mean_pred:.3f}',ha='center',fontsize=10.3,color=INK)
    ax.text(.63,4.26,f'{title}  ·  residual differences after symmetry alignment',fontsize=13.0,color=INK)
    ax.text(8.50,2.01,'mean |Δp|',ha='center',fontsize=9,color=SUB)
    ax.text(.67,.16,'Tile color: RMS activation discrepancy on held-out observations    ·    edge hue: sign of Δweight',fontsize=8.8,color=SUB)
    cax=fig.add_axes([.65,.14,.21,.018]);sm=plt.cm.ScalarMappable(norm=Normalize(0,vmax),cmap='magma')
    cb=fig.colorbar(sm,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=7);cb.set_label('RMS activation discrepancy',fontsize=7)
    _export(fig,Path(directory)/'residual_atlas')


def plot_alignment_diagnostic(modelA,modelB,alignedB,X_align,X_test,y_test,title,metrics,directory):
    from sklearn.metrics import roc_auc_score
    before=activations(modelB.coefs_,modelB.intercepts_,X_test)[-1]
    after=activations(alignedB.coefs,alignedB.intercepts,X_test)[-1]
    ppA=activations(modelA.coefs_,modelA.intercepts_,X_test)[-1]
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.6),dpi=200)
    for ax in axs:
        ax.spines[['right','top']].set_visible(False);ax.tick_params(labelsize=9)
    ax=axs[0]
    left=np.arange(2)
    ax.plot(left,np.array(metrics['raw_mean_corr_train']),'o--',lw=1.3,markersize=7,color='#bd6549',label='raw neuron order')
    ax.plot(left,np.array(metrics['aligned_mean_corr_train']),'o-',lw=1.7,markersize=7,color='#377fa8',label='symmetry aligned')
    ax.set_xticks(left,['Hidden 1','Hidden 2']);ax.set_ylim(-.3,1.05)
    ax.set_ylabel('Matched neuron activation correlation',fontsize=9)
    ax.legend(frameon=False,fontsize=8,loc='lower left')
    ax=axs[1]
    for label,mask,color in [('class 0',y_test==0,'#427fad'),('class 1',y_test==1,'#dc845d')]:
        ax.scatter(ppA[mask],before[mask],s=27,alpha=.74,color=color,edgecolors='white',linewidths=.45,label=label)
    ax.plot([0,1],[0,1],color='#a8b6be',lw=.9,ls='--')
    ax.set_xlim(-.04,1.04);ax.set_ylim(-.04,1.04)
    ax.set_xlabel('Fit A predicted probability',fontsize=9)
    ax.set_ylabel('Fit B predicted probability',fontsize=9)
    ax.legend(frameon=False,fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.84,wspace=.33)
    fig.text(.09,.94,f'{title}: empirical matching vs predictive agreement',fontsize=12.5,color=INK,weight='medium')
    fig.text(.09,.04,'Neuron matching is learned from training inputs; independent model predictions are not made identical.',fontsize=8.1,color=SUB)
    assert np.max(np.abs(before-after)) < 1e-11
    _export(fig,Path(directory)/'matching_diagnostic')
