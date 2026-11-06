"""A model-specific view for linear regression (not a fake SOM lattice)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt


def plot_coefficients(model_or_coefficients, *, labels=None, figsize=(4.8,3.15),
                      xlabel="coefficient (per standardized input)"):
    """Plot signed fitted coefficients; accepts sklearn coef_ or 1D array.

    Interpretation depends on the fitted input and response scales. No feature
    importance or causality is inferred. Does not represent an arbitrary nonlinear
    regression as though it were linear.
    """
    if hasattr(model_or_coefficients,"coef_"):
        coef=np.asarray(model_or_coefficients.coef_,float)
    else: coef=np.asarray(model_or_coefficients,float)
    if coef.ndim!=1 or not np.isfinite(coef).all():
        raise ValueError("single-output linear coefficients must be a finite 1D vector")
    n=len(coef)
    if labels is None: labels=[f"x{i+1}" for i in range(n)]
    if len(labels)!=n: raise ValueError("one label per coefficient required")
    fig,ax=plt.subplots(figsize=figsize,dpi=210)
    y=np.arange(n)[::-1]
    col=np.where(coef>=0,"#3068BB","#D15D48")
    for v,z,color in zip(coef,y,col):
        ax.plot([0,v],[z,z],color=color,lw=4.3,alpha=.74,solid_capstyle="round")
        ax.scatter([v],[z],s=68,color=color,zorder=5,edgecolors="white",linewidths=.7)
    ax.axvline(0,color="#9AA5B5",linewidth=1)
    ax.set_yticks(y,labels,fontsize=10)
    ax.set_xlabel(xlabel,color="#405068",fontsize=9)
    ax.tick_params(axis="x",labelsize=8,length=0)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=.12,zorder=-3)
    for spine in ax.spines.values():spine.set_visible(False)
    max_abs=max(np.max(np.abs(coef)),1e-5)
    ax.set_xlim(-1.17*max_abs,1.17*max_abs)
    ax.set_ylim(-.7,n-.35)
    fig.tight_layout(pad=1)
    return fig,ax


def plot_predictions(y_true, y_pred, *, x=None, baseline=None,
                     xlabel="Observation index", ylabel="Observed response",
                     figsize=(5.25,2.8)):
    """A minimal time-ordered predictive check; accepts arrays, not a model.

    Shows outcomes on the supplied evaluation set. 'baseline' may be a
    reference prediction sequence. It is deliberately not a confidence band.
    """
    y=np.asarray(y_true,float).reshape(-1)
    pred=np.asarray(y_pred,float).reshape(-1)
    if len(y)!=len(pred) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("y_true and y_pred must be finite and equal length")
    if x is None: x=np.arange(len(y))
    else:
        x=np.asarray(x)
        if x.ndim!=1 or len(x)!=len(y): raise ValueError("x length must match y")
    fig,ax=plt.subplots(figsize=figsize,dpi=215)
    if baseline is not None:
        b=np.asarray(baseline,float).reshape(-1)
        if len(b)!=len(y) or not np.isfinite(b).all(): raise ValueError("baseline must match y")
        ax.plot(x,b,lw=1.0,color="#9AA7B7",alpha=.83,label="Persistence baseline",zorder=1)
    ax.plot(x,y,lw=1.2,color="#26354A",alpha=.72,label="Observed",zorder=3)
    ax.plot(x,pred,lw=1.75,color="#CF583D",alpha=.85,label="Model prediction",zorder=2)
    ax.set_xlabel(xlabel,fontsize=9,color="#344257")
    ax.set_ylabel(ylabel,fontsize=9,color="#344257")
    ax.tick_params(length=0,labelsize=8)
    ax.grid(lw=.65,color="#7D8FA6",alpha=.15)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.legend(frameon=False,fontsize=7.7,ncol=3,loc="upper right")
    fig.tight_layout(pad=.65)
    return fig,ax
