"""Symmetry-aware alignment for independent tanh MLPs.

This module matches *representations*, not functions.  Signed hidden-unit
permutations form an exact reparameterization symmetry for tanh networks.
Matching by hidden activation traces is empirical and depends on the reference
distribution.  Alignment never improves predictions or makes independent fits
functionally equivalent.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit


@dataclass(frozen=True)
class Alignment:
    """Immutable report and transformed copy of a candidate network."""

    coefs: tuple
    intercepts: tuple
    permutations: tuple
    signs: tuple
    raw_mean_abs: tuple
    aligned_mean_abs: tuple
    raw_mean_corr: tuple
    aligned_mean_corr: tuple
    prediction_preservation_error: float


def _validate(coefs, intercepts):
    coefs = tuple(np.asarray(w, dtype=float).copy() for w in coefs)
    intercepts = tuple(np.asarray(b, dtype=float).copy() for b in intercepts)
    if len(coefs) < 2 or len(coefs) != len(intercepts):
        raise ValueError('At least one hidden layer and matching intercepts are required')
    for i, (w, b) in enumerate(zip(coefs, intercepts)):
        if w.ndim != 2 or b.ndim != 1 or w.shape[1] != b.size:
            raise ValueError(f'Invalid layer shapes at index {i}')
        if i and coefs[i-1].shape[1] != w.shape[0]:
            raise ValueError('Incompatible adjacent layers')
    if coefs[-1].shape[1] != 1:
        raise ValueError('Only binary logistic-output MLPs are supported')
    return coefs, intercepts


def activations(coefs, intercepts, X):
    """Return input, tanh hidden activations, logits, and probabilities.

    Assumes a binary logistic output, as in sklearn MLPClassifier with tanh
    activation and two classes. ``X`` must already be transformed exactly as
    during fitting.
    """
    coefs, intercepts = _validate(coefs, intercepts)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != coefs[0].shape[0]:
        raise ValueError('X has the wrong number of features')
    layers = [X]
    for w, b in zip(coefs[:-1], intercepts[:-1]):
        layers.append(np.tanh(layers[-1] @ w + b))
    logits = (layers[-1] @ coefs[-1] + intercepts[-1]).ravel()
    return layers, logits, expit(logits)


def transform_signed_permutation(coefs, intercepts, permutations, signs):
    """Apply exact tanh signed-permutation gauge transformation.

    For each hidden layer, ``permutations[l]`` specifies its new order of old
    indices; ``signs[l]`` consists of +/-1 factors on the new activations.
    Incoming weights and biases are transformed, and outgoing weights are
    adjusted so the function is unchanged for *all* inputs.
    """
    coefs, intercepts = _validate(coefs, intercepts)
    if len(permutations) != len(coefs) - 1 or len(signs) != len(permutations):
        raise ValueError('Provide one permutation and sign vector per hidden layer')
    W = [w.copy() for w in coefs]
    b = [x.copy() for x in intercepts]
    for layer, (perm, sign) in enumerate(zip(permutations, signs)):
        n = W[layer].shape[1]
        perm = np.asarray(perm, dtype=int)
        sign = np.asarray(sign, dtype=float)
        if (perm.shape != (n,) or sorted(perm.tolist()) != list(range(n))
            or sign.shape != (n,) or not np.all(np.isin(sign, [-1,1]))):
            raise ValueError('Permutation and signs must match hidden layer width')
        W[layer] = W[layer][:,perm] * sign[None,:]
        b[layer] = b[layer][perm] * sign
        W[layer+1] = W[layer+1][perm,:] * sign[:,None]
    return tuple(W), tuple(b)


def _paired_corr(A,B):
    U=A-A.mean(axis=0,keepdims=True)
    V=B-B.mean(axis=0,keepdims=True)
    scale=np.sqrt(np.sum(U*U,axis=0)*np.sum(V*V,axis=0))
    c=np.divide(np.sum(U*V,axis=0),scale,out=np.zeros(A.shape[1]),where=scale>1e-14)
    return float(np.mean(c))


def match_tanh_mlp(reference, candidate, X_align, X_check=None):
    """Match independently trained MLPs by optimal signed activation pairs.

    Parameters
    ----------
    reference, candidate: sklearn MLPClassifier or (coefs_, intercepts_) pair
        Identical hidden-layer widths and tanh/logistic architecture. They
        need not have identical predictions.
    X_align: array (n_observations, n_features)
        **Training or calibration data only**; do not use held-out test labels
        or held-out test inputs to learn a correspondence.
    X_check: array, optional
        Additional inputs for numerical function-invariance testing.

    Notes
    -----
    A Hungarian assignment minimizes signed mean-squared differences between
    reference and candidate hidden-neuron activation profiles on X_align.
    This alignment is not uniquely defined for duplicated/inactive neurons.
    It does not establish functional equivalence across independently trained
    models, nor does it change candidate predictions.
    """
    def unpack(m):
        if isinstance(m, tuple) and len(m)==2:
            return _validate(m[0],m[1])
        if getattr(m, 'activation',None) != 'tanh' or getattr(m,'out_activation_',None) != 'logistic':
            raise ValueError('Only fitted binary tanh MLPClassifier is supported')
        return _validate(m.coefs_,m.intercepts_)
    refW,refb=unpack(reference)
    canW,canb=unpack(candidate)
    if len(refW)!=len(canW) or [w.shape for w in refW] != [w.shape for w in canW]:
        raise ValueError('Reference and candidate must have matching architecture')
    X_align=np.asarray(X_align,dtype=float)
    ref_act,_,_=activations(refW,refb,X_align)
    can_act,_,p_before=activations(canW,canb,X_align)
    permutations=[];signs=[]
    for Ra,Ca in zip(ref_act[1:],can_act[1:]):
        plus=np.mean((Ra[:,:,None]-Ca[:,None,:])**2,axis=0)
        minus=np.mean((Ra[:,:,None]+Ca[:,None,:])**2,axis=0)
        cost=np.minimum(plus,minus)
        rows,cols=linear_sum_assignment(cost)
        perm=np.empty(Ra.shape[1],dtype=int)
        sg=np.ones(Ra.shape[1])
        perm[rows]=cols
        sg[rows]=np.where(minus[rows,cols]<plus[rows,cols],-1.,1.)
        permutations.append(perm)
        signs.append(sg)
    W_aligned,b_aligned=transform_signed_permutation(canW,canb,permutations,signs)
    aligned_act,_,p_after=activations(W_aligned,b_aligned,X_align)
    errors=[]
    for X in (X_align,X_check):
        if X is not None:
            _,_,p0=activations(canW,canb,X)
            _,_,p1=activations(W_aligned,b_aligned,X)
            errors.append(np.max(np.abs(p0-p1)))
    max_err=float(max(errors))
    if max_err>1e-11:
        raise AssertionError(f'Permutation invariance failed: {max_err}')
    return Alignment(
        coefs=W_aligned,intercepts=b_aligned,
        permutations=tuple(permutations),signs=tuple(signs),
        raw_mean_abs=tuple(float(np.mean(np.abs(a-b))) for a,b in zip(ref_act[1:],can_act[1:])),
        aligned_mean_abs=tuple(float(np.mean(np.abs(a-b))) for a,b in zip(ref_act[1:],aligned_act[1:])),
        raw_mean_corr=tuple(_paired_corr(a,b) for a,b in zip(ref_act[1:],can_act[1:])),
        aligned_mean_corr=tuple(_paired_corr(a,b) for a,b in zip(ref_act[1:],aligned_act[1:])),
        prediction_preservation_error=max_err)


def representational_metrics(reference, candidate, X):
    """Compare aligned hidden-unit activation profiles on independent inputs."""
    a,_,p1=activations(reference.coefs_,reference.intercepts_,X)
    b,_,p2=activations(candidate.coefs,candidate.intercepts,X)
    return dict(
        mean_abs_activation=[float(np.mean(np.abs(x-y))) for x,y in zip(a[1:],b[1:])],
        mean_corr_activation=[_paired_corr(x,y) for x,y in zip(a[1:],b[1:])],
        mean_abs_prediction_difference=float(np.mean(np.abs(p1-p2))),
        max_abs_prediction_difference=float(np.max(np.abs(p1-p2))))
