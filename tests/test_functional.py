import numpy as np
import pytest
from modelatlas.functional import (functional_map,information_lattice,monotonicity_violations,
                                   find_empirical_equivalences,model_predictions)


def test_functional_metric_and_duplicate():
    x=np.array([0,.4,.8,1.])
    y=np.array([0,.2,.4,.6])
    atlas=functional_map({'a':x,'b':y,'c':x.copy()})
    np.testing.assert_allclose(atlas.distances[0,2],0,atol=1e-14)
    np.testing.assert_allclose(atlas.distances[0,1],np.sqrt(np.mean((x-y)**2)),atol=1e-14)
    assert any(len(group)==2 for group in find_empirical_equivalences(atlas))
    assert atlas.variance_retained>.99999 # 3 models, one duplicate => 1D


def test_rms_reference_weights():
    a=functional_map({'a':np.array([0,2]),'b':np.array([1,0])},sample_weights=np.array([1,3]))
    np.testing.assert_allclose(a.distances[0,1],np.sqrt((1+3*4)/4))


def test_information_lattice_is_monotone_and_shapley_complete():
    rng=np.random.default_rng(7)
    X=rng.normal(size=(360,7)); p=1/(1+np.exp(-X@np.arange(1,8)/5))
    lattice=information_lattice(X,p,feature_indices=[0,1,2,3],feature_names=['A','B','C','D'])
    assert len(lattice.nodes)==16
    assert monotonicity_violations(lattice)==[]
    np.testing.assert_allclose(sum(lattice.shapley_variance_reduction),
                               lattice.baseline_variance-lattice.nodes[-1].residual_variance,atol=2e-14)
    assert (lattice.shapley_variance_reduction>=-1e-12).all()


def test_prediction_adapter_callable():
    X=np.ones((12,3))
    pp=model_predictions({'example':lambda x:np.ones(len(x))*.2},X)
    assert pp['example'].shape==(12,)


def test_invalid_reference_and_features():
    with pytest.raises(ValueError):functional_map({'a':[0,np.nan]})
    with pytest.raises(ValueError):information_lattice(np.ones((5,3)),np.ones(5),feature_indices=[0,0])


def test_hidden_neuron_permutation_invariant_predictions():
    """Check a genuine parameter symmetry, not merely copied prediction arrays."""
    import copy
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification

    X,y=make_classification(n_samples=145,n_features=5,n_informative=3,random_state=55)
    model=MLPClassifier(hidden_layer_sizes=(7,4),activation='tanh',solver='lbfgs',
                        alpha=.7,random_state=1,max_iter=250).fit(X,y)
    transformed=copy.deepcopy(model)
    rng=np.random.default_rng(555)
    for layer in range(len(transformed.coefs_)-1):
        perm=rng.permutation(transformed.coefs_[layer].shape[1])
        transformed.coefs_[layer]=transformed.coefs_[layer][:,perm]
        transformed.intercepts_[layer]=transformed.intercepts_[layer][perm]
        transformed.coefs_[layer+1]=transformed.coefs_[layer+1][perm,:]
    preds=model_predictions({'base':model,'permuted':transformed},X)
    np.testing.assert_allclose(preds['base'],preds['permuted'],atol=1e-12,rtol=0)
    atlas=functional_map(preds)
    assert atlas.distances[0,1] < 1e-12
