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
        if (w.ndim != 2 or min(w.shape) < 1 or b.shape != (w.shape[1],)
            or not np.isfinite(w).all() or not np.isfinite(b).all()):
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
    if X.ndim != 2 or not len(X) or X.shape[1] != coefs[0].shape[0] or not np.isfinite(X).all():
        raise ValueError('X must be finite, nonempty, and have the fitted number of features')
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
        perm = np.asarray(perm)
        sign = np.asarray(sign, dtype=float)
        if (perm.shape != (n,) or perm.dtype.kind not in "iu" or sorted(perm.tolist()) != list(range(n))
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


def _tanh_parameters(model):
    if isinstance(model, tuple) and len(model) == 2:
        return _validate(model[0], model[1])
    if getattr(model, "activation", None) != "tanh" or getattr(model, "out_activation_", None) != "logistic":
        raise ValueError("Only fitted binary tanh MLPClassifier is supported")
    return _validate(model.coefs_, model.intercepts_)


def match_tanh_mlp(reference, candidate, X_align, X_check=None):
    """Match independently trained MLPs by optimal signed activation pairs.

    Parameters
    ----------
    reference, candidate
        Fitted binary tanh MLPClassifiers, or pairs of weight and bias
        arrays representing the same architecture.
    X_align
        Fitted-space inputs used to learn the matching. Use training or
        calibration data so held-out examples can assess the comparison.
    X_check
        Additional fitted-space inputs for numerical invariance testing.

    Notes
    -----
    A Hungarian assignment minimizes signed mean-squared differences between
    reference and candidate hidden-neuron activation profiles on X_align.
    This alignment is not uniquely defined for duplicated/inactive neurons.
    It does not establish functional equivalence across independently trained
    models, nor does it change candidate predictions.
    """
    if hasattr(reference, "classes_") and hasattr(candidate, "classes_") and not np.array_equal(
        reference.classes_, candidate.classes_
    ):
        raise ValueError("Reference and candidate must use the same ordered class labels")
    refW,refb= _tanh_parameters(reference)
    canW,canb= _tanh_parameters(candidate)
    if len(refW)!=len(canW) or [w.shape for w in refW] != [w.shape for w in canW]:
        raise ValueError('Reference and candidate must have matching architecture')
    X_align=np.asarray(X_align,dtype=float)
    ref_act,_,_=activations(refW,refb,X_align)
    can_act,_,_=activations(canW,canb,X_align)
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
    aligned_act,_,_=activations(W_aligned,b_aligned,X_align)
    errors=[]
    for X in (X_align,X_check):
        if X is not None:
            _, logits_before, p0 = activations(canW, canb, X)
            _, logits_after, p1 = activations(W_aligned, b_aligned, X)
            if not np.allclose(logits_before, logits_after, rtol=1e-11, atol=1e-11):
                raise AssertionError("Permutation invariance failed for candidate logits")
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
    if [np.shape(w) for w in reference.coefs_] != [w.shape for w in candidate.coefs]:
        raise ValueError("Reference and aligned candidate must have matching architecture")
    a,_,p1=activations(*_tanh_parameters(reference), X)
    b,_,p2=activations(candidate.coefs,candidate.intercepts,X)
    return dict(
        mean_abs_activation=[float(np.mean(np.abs(x-y))) for x,y in zip(a[1:],b[1:])],
        mean_corr_activation=[_paired_corr(x,y) for x,y in zip(a[1:],b[1:])],
        mean_abs_prediction_difference=float(np.mean(np.abs(p1-p2))),
        max_abs_prediction_difference=float(np.max(np.abs(p1-p2))))
