import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import matplotlib.pyplot as plt
from modelatlas import SOMMap, PhotoZCells, sharp_tail_bounds, plot_photoz_map


def _toy():
    w=np.zeros((2,2,2));w[0,1]=[2,0];w[1,0]=[0,2];w[1,1]=[2,2]
    X=np.array([[0,0],[0,0],[0,0],[2,0],[2,0],[0,2],[2,2]],float)
    # two recorded in first cell, one in second, none in third/fourth
    z=np.array([.6,1.4,np.nan,1.1,np.nan,np.nan,np.nan])
    attempted=np.array([1,1,0,1,0,1,0],bool)
    return PhotoZCells(SOMMap(w),X,z,attempted)


def test_identification_exact_unrestricted_and_empty_spectrum():
    m=_toy();b0=m.identify(z0=1.,gamma=0)
    np.testing.assert_allclose([b0.lower[0,0],b0.upper[0,0]],[.5,.5])
    np.testing.assert_allclose([b0.lower[0,1],b0.upper[0,1]],[1.,1.])
    np.testing.assert_allclose([b0.lower[1,0],b0.upper[1,0]],[0.,1.])
    np.testing.assert_allclose([b0.lower[1,1],b0.upper[1,1]],[0.,1.])
    bn=m.identify(z0=1.,gamma=None)
    assert bn.lower[0,0]==pytest.approx(1/3) and bn.upper[0,0]==pytest.approx(2/3)
    assert bn.lower[0,1]==1/2 and bn.upper[0,1]==1
    l,u=bn.population_interval()
    assert l==pytest.approx((1+1)/7)
    assert u==pytest.approx((2+2+1+1)/7)
    assert bn.summary()['N_usable_spectra']==3
    assert bn.summary()['N_attempted']==4
    assert np.isnan(bn.success_given_attempt[1,1])


def test_tail_w1_sharp_budget_cost_order():
    z=np.array([.2,.7,1.1,1.8])
    # At z0=1, p=.5. The cheapest positive-mass move for upper tail costs
    # (1-.7)/4=.075, and for lower (1.1-1)/4=.025.
    assert sharp_tail_bounds(z,1.,0)==(.5,.5)
    l,u=sharp_tail_bounds(z,1.,.025)
    assert l==pytest.approx(.25)
    assert u==pytest.approx(.5+.025/.3)
    l,u=sharp_tail_bounds(z,1.,.075)
    assert l==pytest.approx(.25-(.075-.025)/.8)
    assert u==pytest.approx(.75)
    assert sharp_tail_bounds(z,1.,None)==(0.,1.)
    for gamma in [.01,.1,.5,2]:
        l,u=sharp_tail_bounds(z,1.,gamma)
        assert 0<=l<=.5<=u<=1
    with pytest.raises(ValueError): sharp_tail_bounds(z,1,-.1)


def test_spectroscopy_and_grid_validation():
    m=_toy()
    assert m.identify(1,.1).n_target.sum()==7
    with pytest.raises(ValueError):
        PhotoZCells(m.som,m.target_features,m.spectroscopic_redshift,
                    attempted=np.zeros(7,bool))
    with pytest.raises(ValueError):
        m.identify(1.,-1)
    with pytest.raises(ValueError):
        PhotoZCells(m.som,m.target_features,np.zeros(6))


def test_map_plot_and_hatch():
    b=_toy().identify(1.,.2)
    fig,ax=plot_photoz_map(b,'identified_width',highlight=(0,0))
    assert len(ax.patches)>=12
    plt.close(fig)
    fig,ax=plot_photoz_map(b,'spectroscopic_coverage')
    plt.close(fig)
    with pytest.raises(ValueError):plot_photoz_map(b,'unknown')


def test_additive_width_and_ranking():
    b=_toy().identify(1.,.2)
    pop_l,pop_u=b.population_interval()
    assert np.nansum(b.population_width_contribution)==pytest.approx(pop_u-pop_l)
    ranked=b.rank_cells(top=4)
    assert len(ranked)==4
    assert ranked[0]['population_width_contribution']>=ranked[-1]['population_width_contribution']
