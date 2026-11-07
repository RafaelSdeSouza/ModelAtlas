"""Mathematical checks for signed-permutation symmetry and model comparison."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from modelatlas.symmetry import (
    activations, transform_signed_permutation, match_tanh_mlp,
    representational_metrics,
)
from modelatlas.symmetry_plot import plot_independent_pair


def _fitted(seed=9):
    rng=np.random.default_rng(111)
    X=rng.normal(size=(160,8))
    y=(X[:,0]+.9*X[:,1]*X[:,2]-.15*X[:,4]>0).astype(int)
    m=MLPClassifier(hidden_layer_sizes=(16,9),activation='tanh',random_state=seed,
                    max_iter=160,alpha=.1,learning_rate_init=.008).fit(X,y)
    return X,m


def test_exact_signed_permutation_preserves_function():
    X,net=_fitted()
    rng=np.random.default_rng(44)
    p=[rng.permutation(16),rng.permutation(9)]
    signs=[rng.choice([-1,1],16),rng.choice([-1,1],9)]
    W,B=transform_signed_permutation(net.coefs_,net.intercepts_,p,signs)
    pred1=activations(net.coefs_,net.intercepts_,X)[-1]
    pred2=activations(W,B,X)[-1]
    assert np.max(np.abs(pred1-pred2))<1e-12


def test_alignment_recovers_exact_gauge_without_changing_prediction():
    X,net=_fitted()
    rng=np.random.default_rng(145)
    p=[rng.permutation(16),rng.permutation(9)]
    signs=[rng.choice([-1,1],16),rng.choice([-1,1],9)]
    W,B=transform_signed_permutation(net.coefs_,net.intercepts_,p,signs)
    fit=match_tanh_mlp(net,(W,B),X[:115],X_check=X[115:])
    assert max(fit.aligned_mean_abs)<1e-11
    assert max(abs(c-1) for c in fit.aligned_mean_corr)<1e-11
    assert fit.prediction_preservation_error<1e-12


def test_independent_fits_preserve_only_candidate_function():
    X,ref=_fitted(10)
    _,candidate=_fitted(30)
    fit=match_tanh_mlp(ref,candidate,X[:115],X_check=X[115:])
    pA=activations(ref.coefs_,ref.intercepts_,X)[-1]
    pB=activations(candidate.coefs_,candidate.intercepts_,X)[-1]
    aligned=activations(fit.coefs,fit.intercepts,X)[-1]
    assert np.max(abs(pB-aligned))<1e-12
    assert np.max(abs(pA-pB))>1e-5
    assert sum(fit.aligned_mean_abs)<sum(fit.raw_mean_abs)
    met=representational_metrics(ref,fit,X[115:])
    assert met['max_abs_prediction_difference']>0


def test_bad_signs_rejected():
    X,net=_fitted()
    with pytest.raises(ValueError):
        transform_signed_permutation(net.coefs_,net.intercepts_,
                                     [np.arange(16),np.arange(9)],
                                     [np.ones(16)*2,np.ones(9)])


def test_plot_export(tmp_path):
    X,ref=_fitted(10)
    _,candidate=_fitted(30)
    fit=match_tanh_mlp(ref,candidate,X[:115],X_check=X[115:])
    pA=activations(ref.coefs_,ref.intercepts_,X)[-1]
    pB=activations(candidate.coefs_,candidate.intercepts_,X)[-1]
    row={ 'test_auc_A':.8,'test_auc_B':.8,'seed_A':10,'seed_B':30,
          'selected_p_A':float(pA[-1]),'selected_p_B':float(pB[-1])}
    plot_independent_pair(ref,fit,X[-1],'Toy example','class 1',row,tmp_path)
    for ext in ('png','svg','pdf'):
        assert (tmp_path/('independent_networks_aligned.'+ext)).stat().st_size>2000
