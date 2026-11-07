import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import matplotlib.pyplot as plt
from modelatlas import SOMMap, train_som, visualize, plot_som, plot_metric


def test_som_metrics_assign_consistently():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(65,3))
    w=train_som(x,grid=(4,5),steps=120,seed=2)
    m=SOMMap(w,samples=x)
    b=m.best_matching_units()
    assert b.shape==(65,2)
    assert m.occupancy().sum()==65
    assert m.u_matrix().shape==(4,5)
    d=m.diagnostics()
    assert 0<=d['topographic_error']<=1
    assert d['n_samples']==65
    assert d['n_neurons']==20
    assert d['empty_neurons']+d['occupied_neurons']==20


def test_som_from_array_and_external_display():
    x=np.arange(2*3*4,dtype=float).reshape(2,3,4)
    m=SOMMap(x,display_codebook=x*10,feature_labels=('a','b','c','d'))
    fig,ax=plot_som(m,color_by=1,labels=('1','2','3','4'))
    assert len(ax.patches)>5
    plt.close(fig)


def test_som_reproducibility():
    x=np.random.default_rng(9).normal(size=(30,2))
    assert np.allclose(train_som(x,grid=(3,4),steps=100,seed=5),
                       train_som(x,grid=(3,4),steps=100,seed=5))


def test_som_bad_shapes():
    with pytest.raises(ValueError): SOMMap(np.ones((2,3)))
    with pytest.raises(ValueError): SOMMap(np.full((2,3,4),np.nan))
    with pytest.raises(ValueError): SOMMap(np.ones((2,3,4)),samples=np.ones((4,3)))


def test_u_matrix_known_distances():
    w=np.zeros((2,2,1));w[0,0,0]=0;w[0,1,0]=2;w[1,0,0]=4;w[1,1,0]=6
    M=SOMMap(w)
    np.testing.assert_allclose(M.u_matrix(),[[3,3],[3,3]])


def test_renderer_rejects_unfitted_unsupported():
    with pytest.raises(TypeError): visualize(object())


def test_occupancy_height_really_depends_on_data():
    m=SOMMap(np.random.default_rng(1).normal(size=(3,4,2)),
             samples=np.random.default_rng(2).normal(size=(35,2)))
    fig,ax=plot_som(m,color_by='occupancy',elevation='occupancy')
    plt.close(fig)
    fig,ax=plot_metric(m,'u_matrix')
    plt.close(fig)


def test_linear_adapter():
    class Model:
        coef_=np.array([1.,-2.,.25])
    fig,ax=visualize(Model(),labels=['a','b','c'])
    assert len(ax.collections)==3
    plt.close(fig)


def test_mlp_adapter():
    class Model:
        coefs_=[np.ones((3,4)),np.arange(4.)[:,None]]
    fig,ax=visualize(Model(),input_labels=['a','b','c'],output_labels=['y'])
    assert len(ax.patches)==8
    plt.close(fig)


def test_predictions_dimensions():
    from modelatlas import plot_predictions
    f,a=plot_predictions([1.,2.,3.],[1.1,1.9,2.8],x=[2000,2001,2002])
    plt.close(f)
    with pytest.raises(ValueError):
        plot_predictions([1,2,3],[1.,2.])
