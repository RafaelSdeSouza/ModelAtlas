"""Model-family-agnostic information geometry of probabilistic classifiers.

The reference dataset is part of the distance definition. Bootstrap graph
stability resamples *reference observations*, not fitted model parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ProbabilityAtlas:
    names: tuple[str, ...]
    unique_names: tuple[str, ...]
    equivalence_classes: tuple[tuple[str, ...], ...]
    distances: np.ndarray
    coordinates: np.ndarray
    retained_3d: float
    distance_stress: float
    edge_frequencies: np.ndarray
    geometry: str
    n_reference: int
    n_bootstrap: int


def _validate(predictions: Mapping[str, np.ndarray]):
    names=tuple(predictions)
    if not names: raise ValueError('at least one model required')
    values=[np.asarray(predictions[k], dtype=float) for k in names]
    if any(v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() for v in values):
        raise ValueError('predictions must be finite vectors with at least two observations')
    if len({len(v) for v in values})!=1:
        raise ValueError('all models must share exactly the same reference observations')
    P=np.stack(values)
    if (P<0).any() or (P>1).any():
        raise ValueError('predictive probabilities must be in [0,1]')
    return names, P


def _coords_metric(P, geometry):
    if geometry == 'fisher_rao':
        return 2*np.arcsin(np.sqrt(P))  # Bernoulli Fisher-Rao arc-length coordinate
    if geometry == 'probability_rms':
        return P
    if geometry == 'hellinger':
        # Hellinger chord coordinates, two per Bernoulli measurement
        return np.concatenate([np.sqrt(P),np.sqrt(1-P)],axis=1)
    raise ValueError("geometry must be 'fisher_rao', 'probability_rms', or 'hellinger'")


def _metric_distances(Z, weights):
    # Hellinger already has paired components; n columns is twice n_obs.
    delta=Z[:,None,:]-Z[None,:,:]
    return np.sqrt(np.einsum('mno,o->mn',delta*delta,weights,optimize=True))


def _knn_edges(D,k):
    n=D.shape[0];mask=np.zeros((n,n),dtype=bool)
    for i in range(n):
        order=np.argsort(np.where(np.arange(n)==i,np.inf,D[i]),kind='mergesort')
        for j in order[:min(k,n-1)]: mask[i,j]=True; mask[j,i]=True
    return mask


def build_probability_atlas(predictions:Mapping[str,np.ndarray], *, geometry='fisher_rao',
                            k_neighbors=2, n_bootstrap=250, seed=73,
                            duplicate_tolerance=1e-10, reference_weights=None) -> ProbabilityAtlas:
    """Embed probabilistic predictions and estimate bootstrap kNN-edge stability.

    Under Fisher-Rao, the per-observation distance between Bernoulli(p) and
    Bernoulli(q) is ``2|asin(sqrt(p))-asin(sqrt(q))|``. For models we compute
    its RMS under the empirical reference measure.

    Exact/near duplicates on the *finite reference* are collapsed before
    building the graph; this is not a proof of global functional equivalence.
    Nearest-neighbor edge frequencies quantify dependence on the sampled
    reference distribution. They do not quantify refit or attribution stability.
    """
    names,P=_validate(predictions)
    if len(names)<2: raise ValueError('at least two models required')
    if not (1<=k_neighbors<len(names)) or n_bootstrap<1:
        raise ValueError('invalid k_neighbors or n_bootstrap')
    classes=[];representatives=[]
    for j,name in enumerate(names):
        matches=[i for i,q in enumerate(representatives)
                 if np.max(np.abs(P[j]-P[q])) <= duplicate_tolerance]
        if matches: classes[matches[0]].append(name)
        else:representatives.append(j);classes.append([name])
    if len(representatives)<2:raise ValueError('need at least two distinct reference prediction profiles')
    U=P[representatives]
    n=U.shape[1]
    if reference_weights is None:
        w=np.full(n,1/n)
    else:
        w=np.asarray(reference_weights,dtype=float)
        if w.shape!=(n,) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
            raise ValueError('reference_weights must be finite nonnegative values of shape (n_reference,)')
        w=w/w.sum()
    Z=_coords_metric(U,geometry)
    W=np.concatenate([w/2,w/2]) if geometry=='hellinger' else w
    D=_metric_distances(Z,W)
    center=Z-Z.mean(axis=0,keepdims=True)
    V=center*np.sqrt(W)[None,:]
    _,s,v=np.linalg.svd(V,full_matrices=False)
    coords=np.zeros((len(U),3))
    axes=min(3,len(s))
    coords[:,:axes]=V@v[:axes].T
    for j in range(axes):
        i=int(np.argmax(np.abs(coords[:,j])))
        if coords[i,j]<0:coords[:,j]*=-1
    d3=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=2)
    retained=float(np.sum(s[:3]**2)/np.sum(s**2)) if np.sum(s**2)>0 else 1.
    stress=float(np.linalg.norm(D-d3)/np.linalg.norm(D)) if np.linalg.norm(D)>0 else 0.
    rng=np.random.default_rng(seed)
    counts=np.zeros((len(U),len(U)),float)
    for _ in range(n_bootstrap):
        sample=(rng.integers(0,n,n) if reference_weights is None
                else rng.choice(n,size=n,replace=True,p=w))
        if geometry=='hellinger':
            # coordinates are stacked per class: original positions [0:n], [n:2n]
            sub=np.concatenate([sample,sample+n])
        else:
            sub=sample
        Db=_metric_distances(Z[:,sub],np.full(len(sub),1/len(sub)))
        counts += _knn_edges(Db,k_neighbors)
    counts/=n_bootstrap
    return ProbabilityAtlas(names,tuple(names[i] for i in representatives),
                            tuple(tuple(c) for c in classes),D,coords,
                            retained,stress,counts,geometry,n,n_bootstrap)
