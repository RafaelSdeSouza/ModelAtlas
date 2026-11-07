"""Reproducible prototype graphs and a small GTM baseline.

Research-preview implementations. Neither algorithm uses redshifts to fit its
photometric representation. Input features must be standardized consistently.

References
----------
Fritzke (1995), A Growing Neural Gas Network Learns Topologies.
Bishop, Svensen & Williams (1998), The Generative Topographic Mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.special import logsumexp, softmax

from .photoz import _validate_partition, _validate_sensitivity, sharp_tail_bounds


def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 4 or x.shape[1] < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite array (n>=4, d>=2)")
    return x


def _sqdist(x, w):
    return np.maximum(np.sum(x*x, axis=1)[:, None]
                      + np.sum(w*w, axis=1)[None, :]
                      - 2.0 * x @ w.T, 0.0)


def _prototype_distances(x, prototypes):
    observations = np.asarray(x, dtype=float)
    if (
        observations.ndim != 2 or observations.shape[1] != prototypes.shape[1]
        or not np.isfinite(observations).all()
    ):
        raise ValueError("observations must be finite and match the fitted feature dimension")
    return _sqdist(observations, prototypes)


@dataclass
class PrototypeGraph:
    """Unsupervised prototypes with learned (non-grid) connections."""
    prototypes: np.ndarray
    adjacency: np.ndarray
    errors: np.ndarray
    method: str = "growing-neural-gas"

    def __post_init__(self):
        w = np.asarray(self.prototypes, float)
        a = np.asarray(self.adjacency, bool)
        if w.ndim != 2 or min(w.shape) < 1 or a.shape != (len(w), len(w)) or not np.isfinite(w).all():
            raise ValueError("Invalid prototypes or adjacency")
        if np.any(a != a.T) or np.any(np.diag(a)):
            raise ValueError("adjacency must be undirected and loop-free")
        self.prototypes = w
        self.adjacency = a
        self.errors = np.asarray(self.errors, float)
        if self.errors.shape != (len(w),) or not np.isfinite(self.errors).all() or np.any(self.errors < 0):
            raise ValueError("errors must be finite and nonnegative, one per prototype")

    def assign(self, x):
        return np.argmin(_prototype_distances(x, self.prototypes), axis=1)

    @property
    def edges(self):
        return np.column_stack(np.where(np.triu(self.adjacency, k=1)))


def growing_neural_gas(x, max_nodes=64, steps=16000, insertion_interval=240,
                       winner_lr=0.065, neighbor_lr=0.007, max_edge_age=65,
                       error_decay=0.995, seed=31):
    """Small online Growing Neural Gas (GNG) with competitive Hebbian edges.

    Follows the Fritzke algorithm: adapt winner and neighbors, age edges,
    connect two nearest units, remove old edges, insert units at high-error
    locations, and decay accumulated errors. Fixed seed is deterministic.
    This is a teaching/research implementation, not a production optimizer.
    """
    x = _matrix(x)
    max_nodes, steps, insertion_interval, max_edge_age = map(index, (max_nodes, steps, insertion_interval, max_edge_age))
    if not (0 < winner_lr <= 1 and 0 <= neighbor_lr <= 1 and 0 < error_decay <= 1 and max_edge_age >= 0):
        raise ValueError("invalid GNG learning rates, decay, or edge age")
    if max_nodes < 3 or steps < insertion_interval or insertion_interval < 2:
        raise ValueError("Invalid GNG capacity or training schedule")
    rng = np.random.default_rng(seed)
    w = x[rng.choice(len(x), 2, replace=False)].copy()
    ages = np.full((max_nodes, max_nodes), -1, int)
    errors = np.zeros(max_nodes, float)
    count = 2
    for t in range(1, steps + 1):
        v = x[rng.integers(len(x))]
        ds = np.einsum('ij,ij->i', w[:count]-v, w[:count]-v)
        chosen = np.argpartition(ds, 1)[:2]
        s, q = (int(chosen[0]), int(chosen[1])) if ds[chosen[0]] < ds[chosen[1]] else (int(chosen[1]), int(chosen[0]))
        errors[s] += ds[s]
        neighbors = np.flatnonzero(ages[s, :count] >= 0)
        ages[s, neighbors] += 1
        ages[neighbors, s] += 1
        w[s] += winner_lr * (v-w[s])
        if len(neighbors):
            w[neighbors] += neighbor_lr * (v-w[neighbors])
        ages[s,q] = ages[q,s] = 0
        bad = ages[:count,:count] > max_edge_age
        ages[:count,:count][bad] = -1
        # Protect the winner-neighbor graph against permanently isolated units:
        # Fritzke removes zero-degree neurons, which requires index management;
        # this compact variant reconnects an isolated vertex to its nearest unit.
        deg = np.sum(ages[:count,:count] >= 0, axis=1)
        for ii in np.flatnonzero(deg == 0):
            others = np.sum((w[:count]-w[ii])**2, axis=1)
            others[ii] = np.inf
            jj = int(np.argmin(others))
            ages[ii,jj] = ages[jj,ii] = 0
        if t % insertion_interval == 0 and count < max_nodes:
            # Insert a prototype between high accumulated error nodes.
            a = int(np.argmax(errors[:count]))
            nn = np.flatnonzero(ages[a,:count] >= 0)
            b = int(nn[np.argmax(errors[nn])]) if len(nn) else int(np.argsort(errors[:count])[-2])
            w = np.vstack([w, (w[a] + w[b])*0.5])
            ages[a,b] = ages[b,a] = -1
            ages[a,count] = ages[count,a] = 0
            ages[b,count] = ages[count,b] = 0
            errors[a] *= 0.5
            errors[b] *= 0.5
            errors[count] = errors[a]
            count += 1
        errors[:count] *= error_decay
    return PrototypeGraph(w, ages[:count,:count] >= 0, errors[:count])


@dataclass
class GTMModel:
    """Regularized generative topographic mapping with isotropic residuals."""
    prototypes: np.ndarray
    latent: np.ndarray
    beta: float
    basis: np.ndarray
    regression_weights: np.ndarray
    loss_history: np.ndarray

    def responsibilities(self, x):
        """Posterior latent-cell probabilities under uniform mixture weights."""
        return softmax(-0.5 * self.beta * _prototype_distances(x, self.prototypes), axis=1)

    def assign(self, x):
        """Return MAP latent cells as row-major prototype indices."""
        return np.argmin(_sqdist(np.asarray(x,float), self.prototypes), axis=1)


def train_gtm(x, grid=(8,8), rbf_grid=(4,4), n_iter=70, ridge=0.005, seed=31):
    """GTM EM baseline, fixed latent grid and Gaussian RBF mapping.

    Soft assignments are used for fitting and hard MAP assignments for the
    downstream *comparison only*. Ridge regularization stabilizes the M step.
    Beta is isotropic precision, constrained to a finite range. Initialization
    uses SVD and is deterministic. The legacy seed argument has no effect.
    """
    x = _matrix(x)
    rows, cols = map(index, grid)
    p, q = map(index, rbf_grid)
    n_iter = index(n_iter)
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if min(rows,cols,p,q) < 2 or n_iter < 1: raise ValueError('bad GTM grid')
    rr, cc = np.meshgrid(np.linspace(-1,1,rows),np.linspace(-1,1,cols),indexing='ij')
    latent = np.c_[rr.ravel(),cc.ravel()]
    aa, bb = np.meshgrid(np.linspace(-1,1,p),np.linspace(-1,1,q),indexing='ij')
    centers = np.c_[aa.ravel(),bb.ravel()]
    sig = 1.0 / max(p-1,q-1) * 1.15
    phi = np.exp(-np.sum((latent[:,None,:]-centers[None,:,:])**2,axis=2)/(2*sig*sig))
    phi = np.c_[np.ones(len(phi)),phi]
    mean = x.mean(axis=0)
    _,sing,vt = np.linalg.svd(x-mean, full_matrices=False)
    scale = sing[:2] / np.sqrt(len(x))
    init_map = mean + 0.8 * (latent * scale[None,:]) @ vt[:2,:]
    w = np.linalg.lstsq(phi,init_map,rcond=1e-7)[0]
    prototypes = phi@w
    beta = 1.0
    history=[]
    for _ in range(n_iter):
        distances = _sqdist(x,prototypes)
        lp = -0.5*beta*distances
        lp -= logsumexp(lp,axis=1)[:,None]
        r = np.exp(lp)
        g = r.sum(axis=0)
        lhs = phi.T@(g[:,None]*phi) + ridge*np.eye(phi.shape[1])
        rhs = phi.T@(r.T@x)
        w = np.linalg.solve(lhs,rhs)
        prototypes = phi@w
        newdist = _sqdist(x,prototypes)
        variance = np.sum(r*newdist) / (len(x)*x.shape[1])
        beta = 1. / np.clip(variance, 0.03, 10.0)
        history.append(float(np.mean(logsumexp(-.5*beta*newdist,axis=1)-np.log(len(prototypes))
                                 +.5*x.shape[1]*(np.log(beta)-np.log(2*np.pi)))))
    return GTMModel(prototypes,latent,float(beta),phi,w,np.asarray(history))


def partition_tail_bounds(assignments, spec_z, n_nodes, z0=1.2, gamma=0.16):
    """Apply the existing sharp-tail computation to ANY hard partition.

    No truth redshifts are used: spec_z is NaN unless *usable spectroscopy*.
    Uses independent cellwise W1 restrictions, with empirical spectra fixed.
    gamma=None means no assumption, gamma=0 exact transfer.
    Outputs are *identification* intervals, not confidence intervals.
    """
    n_nodes = index(n_nodes)
    _validate_sensitivity(z0, gamma)
    ids, z = _validate_partition(assignments, n_nodes, spec_z)
    if z is None:
        raise ValueError("spec_z is required; use NaN for missing spectra")
    n=np.bincount(ids,minlength=n_nodes)
    ok=np.isfinite(z)
    s=np.bincount(ids[ok],minlength=n_nodes)
    lo=np.full(n_nodes,np.nan)
    hi=np.full(n_nodes,np.nan)
    for i in np.flatnonzero(n):
        if not s[i]:
            lo[i],hi[i]=0.,1.
            continue
        vals=z[(ids==i)&ok]
        q=s[i]/n[i]
        obs=np.mean(vals>z0)
        a,b=sharp_tail_bounds(vals,z0,gamma)
        lo[i],hi[i]=q*obs+(1.-q)*a,q*obs+(1.-q)*b
    weights=n/n.sum()
    lower=float(np.nansum(weights*lo))
    upper=float(np.nansum(weights*hi))
    return {'lower':lower,'upper':upper,'width':upper-lower,'n_target':n,
            'n_spec':s,'cell_lower':lo,'cell_upper':hi,
            'n_empty_spec':int(np.count_nonzero((n>0)&(s==0)))}