import numpy as np
import pytest
from modelatlas.adaptive import growing_neural_gas,train_gtm,partition_tail_bounds

def _toy():
    r=np.random.default_rng(2)
    return np.r_[r.normal(loc=-1,size=(70,3)),r.normal(loc=1,size=(70,3))]

def test_gng_edges_and_repeatability():
    x=_toy()
    g=growing_neural_gas(x,max_nodes=12,steps=480,insertion_interval=40,seed=4)
    h=growing_neural_gas(x,max_nodes=12,steps=480,insertion_interval=40,seed=4)
    assert g.prototypes.shape==(12,3)
    assert np.array_equal(g.adjacency,g.adjacency.T)
    assert not np.any(np.diag(g.adjacency))
    assert np.allclose(g.prototypes,h.prototypes)
    assert np.count_nonzero(g.adjacency)>0
    assert np.all((g.assign(x)>=0)&(g.assign(x)<12))

def test_gtm_em_shapes():
    x=_toy()
    m=train_gtm(x,grid=(4,4),rbf_grid=(3,3),n_iter=8)
    assert m.prototypes.shape==(16,3)
    assert m.latent.shape==(16,2)
    assert np.isfinite(m.prototypes).all() and m.beta>0
    assert np.isfinite(m.loss_history).all()
    assert np.all((m.assign(x)>=0)&(m.assign(x)<16))

def test_partition_bounds_no_truth_leakage():
    ids=np.array([0,0,0,1,1,1]);z=np.array([.3,np.nan,np.nan,1.4,np.nan,np.nan])
    b=partition_tail_bounds(ids,z,2,z0=1.2,gamma=None)
    assert b['lower']==pytest.approx(1./6)
    assert b['upper']==pytest.approx(5./6)
    b0=partition_tail_bounds(ids,z,2,z0=1.2,gamma=0)
    assert b0['lower']==b0['upper']==pytest.approx(.5)
    assert b['n_empty_spec']==0

def test_partition_empty_spectra_unrestricted():
    ids=np.array([0,0,1,1]);z=np.array([.3,np.nan,np.nan,np.nan])
    b=partition_tail_bounds(ids,z,2,z0=1.2,gamma=0)
    assert b['lower']==0 and b['upper']==.5
    assert b['n_empty_spec']==1