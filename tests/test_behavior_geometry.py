import numpy as np
import pytest
from modelatlas.behavior_geometry import build_probability_atlas


def test_bernoulli_fisher_rao_distance():
    a=np.array([.25,.5,.8]);b=np.array([.6,.2,.4]);
    atlas=build_probability_atlas({'a':a,'b':b},k_neighbors=1,n_bootstrap=12)
    expected=np.sqrt(np.mean((2*np.arcsin(np.sqrt(a))-2*np.arcsin(np.sqrt(b)))**2))
    assert np.isclose(atlas.distances[0,1],expected)
    assert atlas.coordinates.shape==(2,3)
    assert np.isclose(atlas.distance_stress,0,atol=1e-12)


def test_duplicate_predictions_collapse_and_bootstrap_consistency():
    rng=np.random.default_rng(2)
    p=rng.uniform(.1,.9,80)
    a=np.clip(p+.09,0,1)
    atlas=build_probability_atlas({'m':p,'m2':p.copy(),'another':a},n_bootstrap=20)
    assert atlas.equivalence_classes[0]==('m','m2')
    assert len(atlas.unique_names)==2
    assert np.all((atlas.edge_frequencies>=0)&(atlas.edge_frequencies<=1))


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        build_probability_atlas({'a':[-.2,.4],'b':[.5,.9]})


def test_reference_weights_change_fisher_distance():
    a=np.array([.2,.2,.3]);b=np.array([.7,.2,.8]);weights=np.array([10.,1.,1.])
    atlas=build_probability_atlas({'a':a,'b':b},k_neighbors=1,n_bootstrap=8,
                                  reference_weights=weights)
    d=(2*np.arcsin(np.sqrt(a))-2*np.arcsin(np.sqrt(b)))**2
    assert np.isclose(atlas.distances[0,1],np.sqrt(np.dot(weights,d)/weights.sum()))


def test_hellinger_metric():
    a=np.array([.2,.6,.4]);b=np.array([.7,.8,.1])
    atlas=build_probability_atlas({'a':a,'b':b},geometry='hellinger',k_neighbors=1,n_bootstrap=8)
    H2=.5*((np.sqrt(a)-np.sqrt(b))**2+(np.sqrt(1-a)-np.sqrt(1-b))**2)
    assert np.isclose(atlas.distances[0,1],np.sqrt(np.mean(H2)))
