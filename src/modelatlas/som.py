"""SOM fitting and diagnostics. All distances are in fitted feature space.

The small fitter is for reproducible examples, not intended to compete with
production SOM trainers like kohonen or TorchSOM. Users may pass ANY fitted
codebook with shape (rows, columns, features) to SOMMap.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _arr3(w):
    w = np.asarray(w, dtype=float)
    if w.ndim != 3 or min(w.shape) < 1 or not np.isfinite(w).all():
        raise ValueError("SOM codebook must be finite (rows, cols, features)")
    if min(w.shape[:2]) < 2:
        raise ValueError("SOM grid must have at least 2 rows and 2 columns")
    return w


@dataclass(frozen=True)
class SOMMap:
    """Trained SOM map and optional calibration observations.

    codebook: (rows, cols, features) in the exact fitted feature space.
    samples: (n, features) in the same feature space; never raw/unscaled data.
    display_codebook: same shape in user-defined physical units (optional).
    """
    codebook: np.ndarray
    samples: np.ndarray | None = None
    feature_labels: tuple[str, ...] | None = None
    display_codebook: np.ndarray | None = None
    feature_units: tuple[str, ...] | None = None

    def __post_init__(self):
        w = _arr3(self.codebook)
        object.__setattr__(self, "codebook", w)
        if self.samples is not None:
            x = np.asarray(self.samples, dtype=float)
            if x.ndim != 2 or x.shape[1] != w.shape[-1] or not np.isfinite(x).all():
                raise ValueError("samples must be finite (n, n_features), matching codebook")
            object.__setattr__(self, "samples", x)
        if self.display_codebook is not None:
            dw = np.asarray(self.display_codebook, dtype=float)
            if dw.shape != w.shape or not np.isfinite(dw).all():
                raise ValueError("display_codebook must have codebook shape and finite values")
            object.__setattr__(self, "display_codebook", dw)
        if self.feature_labels is not None and len(self.feature_labels) != w.shape[-1]:
            raise ValueError("feature_labels length must equal n_features")
        if self.feature_units is not None and len(self.feature_units) != w.shape[-1]:
            raise ValueError("feature_units length must equal n_features")

    @property
    def rows(self): return self.codebook.shape[0]
    @property
    def cols(self): return self.codebook.shape[1]
    @property
    def n_features(self): return self.codebook.shape[2]

    def distances(self, x=None):
        if x is None:
            if self.samples is None: raise ValueError("observations required for this diagnostic")
            x = self.samples
        a = np.asarray(x, dtype=float)
        if a.ndim != 2 or a.shape[1] != self.n_features:
            raise ValueError("x must be (n, n_features)")
        # nonnegative squared norm calculation; avoids large (n,m,d) allocation
        w = self.codebook.reshape(-1, self.n_features)
        d2 = np.sum(a*a, axis=1)[:,None] + np.sum(w*w, axis=1)[None,:] - 2*a@w.T
        return np.sqrt(np.maximum(d2,0.0))

    def best_matching_units(self, x=None):
        d=self.distances(x)
        flat = np.argmin(d, axis=1)
        return np.column_stack(np.unravel_index(flat, (self.rows,self.cols)))

    def occupancy(self):
        if self.samples is None: raise ValueError("samples required for occupancy")
        bmu=self.best_matching_units()
        return np.bincount(np.ravel_multi_index(bmu.T,(self.rows,self.cols)),
                           minlength=self.rows*self.cols).reshape(self.rows,self.cols)

    def u_matrix(self):
        """Mean Euclidean distance between 4-connected neighboring prototypes."""
        w=self.codebook
        out=np.zeros((self.rows,self.cols),float)
        n=np.zeros((self.rows,self.cols),float)
        for axis in (0,1):
            dif=np.linalg.norm(np.diff(w,axis=axis),axis=2)
            if axis == 0:
                out[:-1,:]+=dif;out[1:,:]+=dif
                n[:-1,:]+=1;n[1:,:]+=1
            else:
                out[:,:-1]+=dif;out[:,1:]+=dif
                n[:,:-1]+=1;n[:,1:]+=1
        return out/n

    def diagnostics(self):
        if self.samples is None: raise ValueError("samples required for diagnostics")
        d=self.distances()
        n=self.samples.shape[0]
        ind=np.argsort(d,axis=1)[:,:2]
        a=np.column_stack(np.unravel_index(ind[:,0],(self.rows,self.cols)))
        b=np.column_stack(np.unravel_index(ind[:,1],(self.rows,self.cols)))
        qe=float(np.mean(d[np.arange(n),ind[:,0]]))
        te=float(np.mean(np.sum(np.abs(a-b),axis=1)!=1))
        o=self.occupancy()
        return {"n_samples":n,"n_neurons":self.rows*self.cols,
                "quantization_error":qe,"topographic_error":te,
                "occupied_neurons":int(np.count_nonzero(o)),
                "empty_neurons":int(np.count_nonzero(o==0)),
                "max_occupancy":int(np.max(o))}


def train_som(x, grid=(7,8), steps=9000, seed=21, learning_rate=.42,
              sigma=None):
    """Fit a rectangular, Euclidean online SOM with a decaying neighborhood.

    x must already have suitable physically justified feature scaling.
    Results are repeatable for fixed inputs and seed. Returns a codebook array.
    """
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)<2 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite (n_samples, n_features) array")
    rows,cols=map(int,grid)
    if min(rows,cols)<2: raise ValueError("grid must be >=2x2")
    if steps<1: raise ValueError("steps must be positive")
    rng=np.random.default_rng(seed)
    # PCA-ish linear initialization for ordered gradients; robust even if rank-deficient
    center=x.mean(axis=0); xc=x-center
    _,_,vt=np.linalg.svd(xc,full_matrices=False)
    basis=np.zeros((2,x.shape[1]))
    basis[:min(2,len(vt))]=vt[:min(2,len(vt))]
    coord0=np.linspace(-1,1,rows)[:,None,None]
    coord1=np.linspace(-1,1,cols)[None,:,None]
    spread=np.std(x,axis=0)
    initial=center + .72*(coord0*basis[0]+coord1*basis[1])*max(np.mean(spread),.25)
    w=np.array(initial,dtype=float,copy=True)
    rr,cc=np.indices((rows,cols))
    if sigma is None: sigma=max(rows,cols)*.54
    for it in range(steps):
        sample=x[rng.integers(len(x))]
        b=np.unravel_index(np.sum((w-sample)**2,axis=-1).argmin(),(rows,cols))
        frac=it/max(steps-1,1)
        lr=learning_rate*(.07/learning_rate)**frac
        rad=sigma*(.72/sigma)**frac
        kernel=np.exp(-((rr-b[0])**2+(cc-b[1])**2)/(2*rad**2))
        w+=lr*kernel[:,:,None]*(sample-w)
    return w


def som_from_minisom(model, X=None, feature_labels=None, display_codebook=None):
    """Adapter for a trained MiniSom object (with get_weights())."""
    if not hasattr(model,"get_weights"):
        raise TypeError("model must provide get_weights()")
    return SOMMap(model.get_weights(),samples=X,feature_labels=feature_labels,
                  display_codebook=display_codebook)
