import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from modelatlas import (SOMMap, PhotoZCells, visualize,
    plot_som_architecture, plot_photoz_architecture)


def _som():
    w=np.array([[[0,1],[1,0]], [[-2,.4],[.2,2.2]]],float)
    return SOMMap(w,feature_labels=('g−r','r−i'))


def test_dense_real_connections_count():
    fig,ax=plot_som_architecture(_som(),highlight=(1,0))
    meta=ax._modelatlas_architecture
    assert meta['features']==2
    assert meta['neurons']==4
    assert meta['connections']==8
    assert meta['highlighted_connections']==2
    assert len(ax.collections)==1
    assert len(ax.collections[0].get_segments())==8
    plt.close(fig)


def test_selected_and_disabled_modes():
    fig,ax=plot_som_architecture(_som(),show_connections='selected',highlight=(0,0))
    assert ax._modelatlas_architecture['connections']==0
    assert ax._modelatlas_architecture['highlighted_connections']==2
    plt.close(fig)
    fig,ax=plot_som_architecture(_som(),show_connections='none',highlight=(0,0))
    assert ax._modelatlas_architecture['connections']==0
    assert ax._modelatlas_architecture['highlighted_connections']==0
    plt.close(fig)


def test_photoz_architecture_and_dispatch():
    M=_som()
    X=np.array([[0,1],[0,1],[1,0],[-2,.4]],float)
    zs=np.array([.8,1.4,1.5,np.nan])
    B=PhotoZCells(M,X,zs).identify(z0=1.2,gamma=None)
    fig,ax=plot_photoz_architecture(B,M,metric='identified_width',
             highlight=(1,0))
    assert ax._modelatlas_architecture['connections']==8
    plt.close(fig)
    fig,ax=visualize(B,kind='photoz',som=M,architecture=True,metric='identified_width')
    assert ax._modelatlas_architecture['neurons']==4
    plt.close(fig)
    fig,ax=visualize(M,architecture=True)
    plt.close(fig)
    with pytest.raises(ValueError): visualize(B,architecture=True)


def test_reject_invalid_arguments():
    M=_som()
    with pytest.raises(ValueError):plot_som_architecture(M,feature_labels=['one'])
    with pytest.raises(ValueError):plot_som_architecture(M,highlight=(9,0))
    with pytest.raises(ValueError):plot_som_architecture(M,show_connections='selected')
    with pytest.raises(ValueError):plot_som_architecture(M,values=np.zeros((3,3)))
    with pytest.raises(ValueError):plot_som_architecture(M,hatch_mask=np.ones((3,3)))
    with pytest.raises(ValueError):plot_som_architecture(np.zeros((2,2,9)))
