"""XAI mathematical contracts, not only rendering smoke tests."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from modelatlas.explanation import explain_mlp,activation_signatures,attribution_top_edges
from modelatlas.explanation_plot import plot_explanation_atlas


@pytest.fixture(scope='module')
def trained():
    rng=np.random.default_rng(4)
    X=rng.normal(size=(150,8))
    y=(X[:,0]-X[:,2]*X[:,3]+.2*rng.normal(size=150)>0).astype(int)
    m=MLPClassifier(hidden_layer_sizes=(16,9),random_state=4,max_iter=250,
                    activation='tanh',learning_rate_init=.008)
    m.fit(X,y)
    return m,X


def test_edge_completeness_each_layer(trained):
    model,X=trained
    e=explain_mlp(model,X[1],baseline=np.zeros(8),steps=256)
    assert len(e.edge_contributions)==3
    assert all(abs(edge.sum()-e.logit_change)<2e-4 for edge in e.edge_contributions)
    assert e.max_layer_residual<2e-4


def test_neuron_conservation(trained):
    model,X=trained
    e=explain_mlp(model,X[3],steps=256)
    assert e.max_node_residual<2e-4
    assert np.allclose(e.node_contributions[0],e.edge_contributions[0].sum(axis=1))


def test_ig_baseline_zero(trained):
    model,X=trained
    e=explain_mlp(model,X[8],baseline=X[8],steps=32)
    assert np.allclose(e.node_contributions[0],0,atol=1e-12)
    assert abs(e.logit_change)<1e-12
    assert abs(e.max_layer_residual)<1e-12


def test_baseline_changes_explanation(trained):
    model,X=trained
    e1=explain_mlp(model,X[2],baseline=np.zeros(8))
    e2=explain_mlp(model,X[2],baseline=np.ones(8)*.3)
    assert not np.allclose(e1.node_contributions[0],e2.node_contributions[0])
    assert e1.max_layer_residual<2e-4 and e2.max_layer_residual<2e-4


def test_repeatability(trained):
    model,X=trained
    e1=explain_mlp(model,X[9],steps=64)
    e2=explain_mlp(model,X[9],steps=64)
    for a,b in zip(e1.edge_contributions,e2.edge_contributions):assert np.array_equal(a,b)


def test_sign_and_probabilities(trained):
    model,X=trained
    e=explain_mlp(model,X[4])
    assert e.probability==pytest.approx(model.predict_proba(X[[4]])[0,1],abs=1e-10)
    assert np.isfinite(e.logit_change)
    assert len(activation_signatures(model,X))==2
    assert len(attribution_top_edges(e))==3


def test_plotting(trained):
    model,X=trained
    e=explain_mlp(model,X[0])
    fig,ax=plot_explanation_atlas(model,e,X_reference=X,feature_names=[f'x{i}' for i in range(8)])
    assert fig is not None and ax is not None
    from matplotlib import pyplot as plt
    plt.close(fig)


def test_invalid_baseline(trained):
    model,X=trained
    with pytest.raises(ValueError):explain_mlp(model,X[0],baseline=np.zeros(7))
    with pytest.raises(ValueError):explain_mlp(model,X[0],steps=0)
