"""Cellwise partial identification for spectroscopic photo-z calibration.

This module does *not* infer missing redshifts from photometry. It conditions on
an already fitted SOM and observed matched target/spectroscopic catalogues.

Assumption for optional sensitivity radius gamma (dimension: redshift):
  W1(P(z | c, no usable spectroscopy), P(z | c, usable spectroscopy)) <= gamma
for each cell c that has a usable spectroscopic redshift.

The empirical usable-spectroscopy distribution is treated as *known*. The bounds
are sharp population compatibility bounds under this idealized restriction,
NOT sampling uncertainty intervals. With no spectroscopy in a populated cell,
the identified interval for theta=P(z>z0|cell) remains [0, 1].
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .som import SOMMap


def _movable_mass(costs: np.ndarray, budget: float, n_total: int) -> float:
    """Maximum source probability mass moved for nonnegative per-unit costs.

    Empirical measure has atoms of mass 1/n_total, each atom can be split.
    The optimal plan moves closest-to-threshold source mass first.
    """
    costs=np.sort(np.asarray(costs,float))
    if not len(costs): return 0.
    if budget <= 0: return 0.
    remaining=float(budget)
    moved=0.
    w=1./n_total
    for c in costs:
        if c <= 1e-14:
            moved+=w
            continue
        part=min(w, remaining/c)
        moved+=part
        remaining-=part*c
        if part < w-1e-14: break
    return min(1.,moved)


def sharp_tail_bounds(z_observed, z0: float, gamma: float | None):
    """Sharp infimum and supremum of P(Z>z0) in a Wasserstein-1 ball.

    W1 uses absolute redshift distance; gamma is in redshift units. The
    reference is the *empirical measure* of successful spectra, treated as
    fixed. gamma=None means no restriction: return (0,1).

    Atoms exactly at the threshold lead to inf/sup that may be approached but
    not attained; at gamma=0, both extrema equal the empirical tail.
    """
    z=np.asarray(z_observed,float)
    if z.ndim!=1 or not len(z) or not np.isfinite(z).all() or (z<0).any():
        raise ValueError('z_observed must be nonempty, finite, nonnegative 1D values')
    if not np.isfinite(z0) or z0<0:
        raise ValueError('z0 must be finite and nonnegative')
    if gamma is None:
        return 0.,1.
    if not np.isfinite(gamma) or gamma<0:
        raise ValueError('gamma must be None or a finite nonnegative W1 radius')
    p=float(np.mean(z>z0))
    if gamma==0: return p,p
    # To decrease tail, move above-threshold mass down to the threshold.
    to_below=_movable_mass(z[z>z0]-z0,gamma,len(z))
    # To increase tail, move at/below-threshold mass up to just above z0.
    to_above=_movable_mass(z0-z[z<=z0],gamma,len(z))
    return max(0.,p-to_below),min(1.,p+to_above)


@dataclass(frozen=True)
class PhotoZBounds:
    """Cellwise identified bounds for the target high-redshift fraction."""
    threshold: float
    gamma: float | None
    n_target: np.ndarray
    n_attempted: np.ndarray
    n_spec: np.ndarray
    observed_high_fraction: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @property
    def coverage(self):
        return np.divide(self.n_spec,self.n_target,
                         out=np.full(self.n_target.shape,np.nan,float),
                         where=self.n_target>0)

    @property
    def attempted_fraction(self):
        return np.divide(self.n_attempted,self.n_target,
                         out=np.full(self.n_target.shape,np.nan,float),
                         where=self.n_target>0)

    @property
    def success_given_attempt(self):
        return np.divide(self.n_spec,self.n_attempted,
                         out=np.full(self.n_target.shape,np.nan,float),
                         where=self.n_attempted>0)

    @property
    def width(self):
        return self.upper-self.lower

    @property
    def population_width_contribution(self):
        """Cell's additive contribution to overall population identified width.

        This is a *current bound-width decomposition*, NOT the expected gain
        from an additional spectrum or an optimal follow-up allocation.
        """
        n=float(self.n_target.sum())
        if n<=0: raise ValueError('empty target')
        return self.n_target / n * self.width

    def rank_cells(self, top=10):
        """Rank cells by *current* contribution to population bound width.

        The ranking omits follow-up success probability, acquisition cost and
        anticipated reduction in Gamma; it is not a decision-theoretic utility.
        """
        a=self.population_width_contribution
        idx=np.argsort(np.nan_to_num(a,nan=-1).ravel())[::-1]
        result=[]
        for flat in idx:
            i,j=np.unravel_index(flat,a.shape)
            if self.n_target[i,j]==0:continue
            result.append({'row':int(i),'col':int(j),
                'n_target':int(self.n_target[i,j]),
                'n_spec':int(self.n_spec[i,j]),
                'width':float(self.width[i,j]),
                'population_width_contribution':float(a[i,j])})
            if len(result)>=top: break
        return result

    @property
    def unidentified(self):
        return (self.n_target>0)&(self.n_spec==0)

    def population_interval(self):
        """Target-weighted identified interval, assuming independent cell balls.

        The bounds add because the constraint on each P_missing,c is independent.
        """
        n=self.n_target.sum()
        if n<1: raise ValueError('no target observations')
        p=self.n_target/n
        return float(np.nansum(p*self.lower)),float(np.nansum(p*self.upper))

    def metric(self,name):
        lookup={'target_occupancy':self.n_target.astype(float),
                'spectroscopic_coverage':self.coverage,
                'attempt_fraction':self.attempted_fraction,
                'success_given_attempt':self.success_given_attempt,
                'identified_width':self.width,
                'population_width_contribution':self.population_width_contribution,
                'lower_bound':self.lower,
                'upper_bound':self.upper,
                'observed_high_fraction':self.observed_high_fraction}
        if name not in lookup: raise ValueError('unknown photo-z metric: '+name)
        return lookup[name]

    def summary(self):
        a,b=self.population_interval()
        return {'N_target':int(self.n_target.sum()),
                'N_attempted':int(self.n_attempted.sum()),
                'N_usable_spectra':int(self.n_spec.sum()),
                'N_populated_cells':int(np.count_nonzero(self.n_target)),
                'N_cells_without_spectra':int(np.count_nonzero(self.unidentified)),
                'z0':float(self.threshold),
                'Gamma_W1_redshift_units':None if self.gamma is None else float(self.gamma),
                'target_tail_lower':a,'target_tail_upper':b,
                'target_tail_width':b-a}


@dataclass(frozen=True)
class PhotoZCells:
    """SOM mapped parent photometry with linked spectral outcomes.

    target_features: (N,d) photometry, in SOM fitted feature coordinates.
    spectroscopic_redshift: (N,) NaN where redshift unavailable; finite >=0
       where a *usable* redshift was obtained. No true redshifts of unobserved
       objects may enter identification computations.
    attempted: optional (N,) bool, identifies targeted spectroscopy attempts.
       If absent, assumed to equal usable-spec status, which precludes
       separate inference about attempt vs success.
    """
    som: SOMMap
    target_features: np.ndarray
    spectroscopic_redshift: np.ndarray
    attempted: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.som,SOMMap):
            raise TypeError('som must be a fitted SOMMap')
        x=np.asarray(self.target_features,float)
        if x.ndim!=2 or x.shape[1]!=self.som.n_features or not len(x) or not np.isfinite(x).all():
            raise ValueError('target_features must be finite (N, n_features)')
        z=np.asarray(self.spectroscopic_redshift,float)
        if z.shape!=(len(x),) or np.any(np.isinf(z)) or np.any(z[np.isfinite(z)]<0):
            raise ValueError('spectroscopic_redshift must be N numbers or NaNs; z>=0')
        hasz=np.isfinite(z)
        attempted=(hasz if self.attempted is None else np.asarray(self.attempted))
        if attempted.shape!=(len(x),) or attempted.dtype.kind!='b':
            raise ValueError('attempted must be a boolean array of length N')
        if np.any(hasz&~attempted):
            raise ValueError('a usable spectroscopic redshift requires attempted=True')
        object.__setattr__(self,'target_features',x)
        object.__setattr__(self,'spectroscopic_redshift',z)
        object.__setattr__(self,'attempted',attempted)

    @property
    def flat_cells(self):
        bmu=self.som.best_matching_units(self.target_features)
        return np.ravel_multi_index(bmu.T,(self.som.rows,self.som.cols))

    def identify(self, z0=1.0, gamma=0.10):
        """Sharp *conditional* identified intervals for target P(z>z0|cell).

        gamma: W1 radius constraining the redshift distribution of galaxies
               without usable spectroscopy relative to the empirical usable
               spectroscopy distribution in each cell. gamma=None removes
               the restriction. gamma=0 is exact within-cell transfer.
        """
        if not np.isfinite(z0) or z0<0: raise ValueError('z0 must be nonnegative')
        if gamma is not None and (not np.isfinite(gamma) or gamma<0):
            raise ValueError('gamma must be None or nonnegative finite')
        ids=self.flat_cells
        ncell=self.som.rows*self.som.cols
        hasz=np.isfinite(self.spectroscopic_redshift)
        count=np.bincount(ids,minlength=ncell)
        attempt=np.bincount(ids[self.attempted],minlength=ncell)
        spec=np.bincount(ids[hasz],minlength=ncell)
        high=np.full(ncell,np.nan)
        lower=np.full(ncell,np.nan);upper=np.full(ncell,np.nan)
        for c in np.flatnonzero(count):
            n=count[c];s=spec[c]
            if s==0:
                lower[c]=0.;upper[c]=1.
                continue
            z=self.spectroscopic_redshift[(ids==c)&hasz]
            q=s/n
            p=float(np.mean(z>z0))
            high[c]=p
            mlo,mhi=sharp_tail_bounds(z,z0,gamma)
            lower[c]=q*p+(1-q)*mlo
            upper[c]=q*p+(1-q)*mhi
        shape=(self.som.rows,self.som.cols)
        return PhotoZBounds(float(z0),gamma,count.reshape(shape),
                            attempt.reshape(shape),spec.reshape(shape),
                            high.reshape(shape),lower.reshape(shape),upper.reshape(shape))
