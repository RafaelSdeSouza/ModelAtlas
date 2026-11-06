"""A compact dense-network view with edges from actual fitted weight matrices."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba


def plot_mlp(model, *, input_labels=None, output_labels=None, max_neurons=20,
             figsize=(4.3,4.1), trim_quantile=.38):
    """Render a fitted sklearn MLP as a compact network.

    Blue/red edge *sign* and stroke width/opacity encode actual fitted weights.
    Weak edges below an abs-weight quantile are omitted for readability and
    the legend marks this threshold. Does not encode activation distributions.
    """
    if not hasattr(model,"coefs_"):
        raise TypeError("a fitted sklearn MLP with coefs_ is required")
    ws=[np.asarray(w,float) for w in model.coefs_]
    widths=[ws[0].shape[0]]+[w.shape[1] for w in ws]
    if any(widths[k]>max_neurons for k in range(len(widths))):
        raise ValueError("network too wide; reduce, summarize or increase max_neurons")
    if not (0<=trim_quantile<1):raise ValueError("trim_quantile must be [0,1)")
    if input_labels is None: input_labels=[f"x{i+1}" for i in range(widths[0])]
    if output_labels is None: output_labels=["ŷ" for _ in range(widths[-1])]
    if len(input_labels)!=widths[0] or len(output_labels)!=widths[-1]:
        raise ValueError("input/output label lengths must match network")
    fig,ax=plt.subplots(figsize=figsize,dpi=230)
    ax.set_axis_off();ax.set_xlim(-.38,1.36);ax.set_ylim(-.14,1.11)
    ax.set_aspect("equal");fig.subplots_adjust(0,0,1,1)
    xpos=np.linspace(.12,1.12,len(widths))
    ys=[np.linspace(.87,.13,num) if num>1 else np.array([.50]) for num in widths]
    for l,w in enumerate(ws):
        v=np.abs(w.ravel());threshold=np.quantile(v,trim_quantile)
        vmax=max(v.max(),1e-10)
        strands=[];colors=[];linewidths=[]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                weight=w[i,j]
                if abs(weight)<threshold: continue
                strands.append([(xpos[l]+.016,ys[l][i]),(xpos[l+1]-.016,ys[l+1][j])])
                frac=abs(weight)/vmax
                colors.append((*to_rgba("#285FA9" if weight>=0 else "#CB6657")[:3],.13+.64*frac))
                linewidths.append(.35+1.2*frac)
        ax.add_collection(LineCollection(strands,colors=colors,linewidths=linewidths,zorder=1))
    for k,(xposk,ycoords) in enumerate(zip(xpos,ys)):
        for j,y in enumerate(ycoords):
            face=("#E5EFFF" if k==0 else "#FFF0E6" if k==len(widths)-1 else "#F4F7FB")
            ax.add_patch(Circle((xposk,y),.016,facecolor=face,edgecolor="#33496B",
                                linewidth=.85,zorder=5))
            if k==0:
                ax.text(xposk-.027,y,input_labels[j],ha="right",va="center",fontsize=9,color="#35445B")
            if k==len(widths)-1:
                ax.text(xposk+.026,y,output_labels[j],ha="left",va="center",fontsize=9,color="#35445B")
    ax.text(.61,.035,"blue: positive     rust: negative     stroke: |weight|",
            fontsize=7.6,color="#637187",ha="center")
    return fig,ax
