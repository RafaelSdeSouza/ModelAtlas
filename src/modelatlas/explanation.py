"""Conservative, additive explanations for fitted binary scikit-learn MLPs.

For an MLP with scalar output logit f, each edge e=(j,k) in layer l
receives the path-integrated contribution

    E[l][j,k] = int_0^1 d f/d z[l,k](x_t) * W[l][j,k]
                         * d a[l-1,j](x_t)/d t dt,
    x_t = baseline + t * (x - baseline).

At every layer, sum(E[l]) = f(x) - f(baseline), up to integration error.
For hidden neuron k, sum of incoming edge contributions is approximately
sum of its outgoing contributions. For inputs, sum outgoing edge
contributions equals Integrated Gradients w.r.t. the input.

Attributions describe a chosen contrast, not causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _activation(z: np.ndarray, kind: str):
    if kind == "tanh":
        a = np.tanh(z)
        return a, 1.0 - a**2
    if kind == "relu":
        return np.maximum(z, 0.0), (z > 0).astype(float)
    if kind == "logistic":
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
        return a, a * (1.0 - a)
    if kind == "identity":
        return z, np.ones_like(z)
    raise ValueError(f"Unsupported hidden activation {kind!r}")


@dataclass(frozen=True)
class PathExplanation:
    """Local, baseline-relative attribution for a scalar logit."""

    input: np.ndarray
    baseline: np.ndarray
    edge_contributions: tuple[np.ndarray, ...]
    node_contributions: tuple[np.ndarray, ...]  # input, hidden..., output
    activations: tuple[np.ndarray, ...]  # input, hidden...
    logit: float
    baseline_logit: float
    probability: float
    baseline_probability: float
    activation_name: str
    quadrature_steps: int

    @property
    def logit_change(self) -> float:
        return self.logit - self.baseline_logit

    @property
    def max_layer_residual(self) -> float:
        return max(abs(e.sum() - self.logit_change) for e in self.edge_contributions)

    @property
    def max_node_residual(self) -> float:
        return max(
            float(np.max(np.abs(self.edge_contributions[i].sum(axis=0)
                                - self.edge_contributions[i+1].sum(axis=1))))
            for i in range(len(self.edge_contributions)-1)
        ) if len(self.edge_contributions)>1 else 0.0


def _verify_network(model):
    if not hasattr(model, 'coefs_') or not hasattr(model, 'intercepts_'):
        raise TypeError('Expected a fitted scikit-learn MLPClassifier with coefs_ and intercepts_.')
    if not getattr(model, 'n_outputs_', 1) == 1:
        raise ValueError('This version supports binary scalar-output MLPs, not multiclass models.')
    weights = tuple(np.asarray(w, dtype=float) for w in model.coefs_)
    biases = tuple(np.asarray(b, dtype=float) for b in model.intercepts_)
    if len(weights) < 2 or weights[-1].shape[1] != 1:
        raise ValueError('At least one hidden layer and one scalar output are required.')
    if any(w.ndim != 2 or len(b) != w.shape[1] for w, b in zip(weights,biases)):
        raise ValueError('Inconsistent model parameters.')
    if getattr(model, 'out_activation_', 'logistic') != 'logistic':
        raise ValueError('Only binary logistic-output MLPClassifier is supported.')
    activation = getattr(model, 'activation', 'tanh')
    if activation not in ('tanh', 'relu', 'logistic', 'identity'):
        raise ValueError('Unsupported hidden activation.')
    return weights, biases, activation


def _forward(weights, biases, activation, x):
    acts=[np.asarray(x, dtype=float)]
    zs=[]
    derivs=[]
    for l, (w,b) in enumerate(zip(weights,biases)):
        z = acts[-1] @ w + b
        zs.append(z)
        if l == len(weights)-1:
            acts.append(z)
            derivs.append(np.ones_like(z))
        else:
            a,d = _activation(z, activation)
            acts.append(a)
            derivs.append(d)
    return acts,zs,derivs


def explain_mlp(model, x: np.ndarray, *, baseline: np.ndarray | None = None,
                steps: int = 256) -> PathExplanation:
    """Explain the change in the *logit* of a fitted binary MLPClassifier.

    With standardized features, an all-zero baseline represents the training
    feature means. A different baseline is legitimate but changes the question.
    Midpoint quadrature is deterministic. ``steps`` controls accuracy.

    Every edge's integrated contribution is conserved through the network.
    Positive attribution raises the output logit relative to baseline.
    """
    weights,biases,activation = _verify_network(model)
    x=np.asarray(x,dtype=float).reshape(-1)
    if baseline is None:
        baseline=np.zeros_like(x)
    else:
        baseline=np.asarray(baseline,dtype=float).reshape(-1)
    if x.shape != (weights[0].shape[0],) or baseline.shape != x.shape:
        raise ValueError('x and baseline must match the input dimension.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(baseline)):
        raise ValueError('Inputs must be finite.')
    if int(steps) != steps or steps < 8:
        raise ValueError('steps must be an integer >= 8.')
    acts,_,_= _forward(weights,biases,activation,x)
    base_acts,_,_=_forward(weights,biases,activation,baseline)
    logit=float(acts[-1][0]);base_logit=float(base_acts[-1][0])
    delta=x-baseline
    edges=[np.zeros_like(w) for w in weights]
    for idx in range(steps):
        t=(idx+0.5)/steps
        a,z,activation_derivs=_forward(weights,biases,activation,baseline+t*delta)
        # Forward tangents d a_l / d t
        derivatives=[delta]
        for l,w in enumerate(weights[:-1]):
            derivatives.append((derivatives[-1] @ w) * activation_derivs[l])
        # Back-propagated derivatives d f / d z_l
        grad_z=[None]*len(weights)
        grad_z[-1]=np.array([1.0])
        for l in range(len(weights)-2,-1,-1):
            grad_z[l]=(weights[l+1] @ grad_z[l+1])*activation_derivs[l]
        # Edge conductance: chain rule decomposition of df / dt
        for l,w in enumerate(weights):
            edges[l] += w * np.outer(derivatives[l],grad_z[l]) / steps
    node=[edges[0].sum(axis=1)]
    node += [e.sum(axis=0) for e in edges]
    return PathExplanation(
        input=x.copy(),baseline=baseline.copy(),
        edge_contributions=tuple(edges),node_contributions=tuple(node),
        activations=tuple(a.copy() for a in acts[:-1]),
        logit=logit,baseline_logit=base_logit,
        probability=_sigmoid(logit),baseline_probability=_sigmoid(base_logit),
        activation_name=activation,quadrature_steps=int(steps)
    )


def activation_signatures(model, X: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return hidden-layer activations for positioning neurons, not attribution."""
    weights,biases,activation=_verify_network(model)
    X=np.asarray(X,dtype=float)
    if X.ndim != 2 or X.shape[1] != weights[0].shape[0]:
        raise ValueError('X must have shape (n_samples, n_features).')
    a=X.copy();hidden=[]
    for w,b in zip(weights[:-1],biases[:-1]):
        z=a @ w+b
        a,_=_activation(z,activation)
        hidden.append(a)
    return tuple(hidden)


def attribution_top_edges(explanation:PathExplanation, n:int=4):
    """Per-layer sorted (input, output, value) tuples, by absolute attribution."""
    result=[]
    for e in explanation.edge_contributions:
        ranks=np.argsort(-np.abs(e).ravel())[:n]
        result.append([(int(i),int(j),float(e[i,j])) for i,j in
                       (np.unravel_index(k,e.shape) for k in ranks)])
    return result
