# Explanation Atlas: scientific and mathematical contract

**Status:** experimental research software, not a validated new XAI method.
The two-observation demo uses measured chemical features in the public UCI Wine
classification dataset, not astronomical observations. The scientific priority
is faithful encoding, testable numerical conservation and reproducibility.

## Models supported by this XAI prototype

Binary `sklearn.neural_network.MLPClassifier` with one scalar logistic output,
fully connected feedforward hidden layers, and hidden activation `tanh`, `relu`,
`logistic` or `identity`. The attribution engine supports arbitrarily many hidden
layers. The *oblique figure renderer* currently supports exactly two hidden
layers, with a maximum of 18 input features and 36/25 hidden units for legibility.
The rest of ModelAtlas has independent SOM, regression and photo-z adapters.

## Attribution target: the logit, not probability

For a fitted classifier, denote its pre-sigmoid scalar output by `f(x)` and the
class probability by `sigmoid(f(x))`. Choose a reference `b` and the line path
`x(t) = b + t(x-b)`, `t ∈ [0,1]`. For standardized data, `b = 0` represents the
feature-wise training means, *not* a scientifically privileged reference.

For layer `l`, the source activations are `a_{l-1}`; destination pre-activations
are `z_l = W_l^T a_{l-1} + bias_l`. An edge from source `j` to destination `k`
receives the path-integrated contribution

    E_l[j,k] = ∫_0^1 [∂f(x(t))/∂z_l,k] W_l[j,k]
                       [d a_{l-1,j}(x(t))/dt] dt.

With an adequate quadrature rule, the following equalities hold:

1. **Layer completeness:** sum over all edges in *each* layer is `f(x)-f(b)`.
2. **Neuron conservation:** total incoming and outgoing edge contribution for
   any hidden neuron are equal when integrated over the same input path.
3. **Integrated gradients:** the sum of contributions leaving input `j` is
   `(x_j-b_j) ∫_0^1 ∂f(x(t))/∂x_j dt`.

The code uses deterministic midpoint quadrature (256 intervals in the demo).
These equalities are numerically tested. The output probability is printed for
intuition, but all colors, line widths and the `Δlogit` annotation refer to the
**logit difference**. Integrated logit contributions do not sum to a difference
in probabilities.

## Visual semantics and limitations

- **Input circles:** signed integrated-gradient contribution, per feature.
- **Connections:** signed path-integrated *edge contributions*, not weights. A
  large raw weight is not necessarily an important edge for this contrast.
- **Hidden neuron tiles:** sum of incoming contributions for each neuron; these
  are conserved to outgoing contributions. Tile color is not neuron activation.
- **White-ringed neurons / thick edges:** a descriptive, greedy high-magnitude
  connected trace. It is **not** a unique causal path or a proof of importance.
- **Neighboring tiles:** a PCA-to-grid matching of neuron activation signatures
  across a reference sample. This is a *layout optimization*, not an intrinsic
  SOM neighborhood or a physical third dimension.
- **Oblique perspective:** purely graphical extrusion. No inferred third spatial
  coordinate is represented by the extrusion.

Caveats: baseline dependence, correlated predictors, gradients near saturated
nonlinearities, model instability, and potential visual selection bias. Feature
attribution explains the fitted model's output contrast, *not* data-generating
causality, point identification or population representativeness.

## The demonstration

- Dataset: `sklearn.datasets.load_wine()` — eight real measured input features.
- Output: Wine category 0 versus the other two wine categories.
- Split: stratified train/test, `random_state=31`.
- Scaler: `StandardScaler`, fitted to training data only.
- MLP: `(8,16,9,1)`, `tanh`, seed 21, regularized with `alpha=0.16`.
- Baseline: zero vector in training-standardized coordinates.
- Figures: two observations evaluated on the same trained network and with the
  same reference-data neuron ordering. Individual colors scale to each figure;
  do not infer comparable absolute magnitudes from color brightness alone.

The data are easy to classify with these features; strong held-out predictive
performance does not validate an XAI visualization.

## Reproducibility

Run `python examples/wine_explanation_atlas.py` from the project root. It
writes PNG, SVG, PDF and a machine-readable `wine_explanation_metrics.json`.
Run `pytest -q` for both old and XAI diagnostics. Versioned releases should pin
model/data versions and preserve these metrics and vector figures.

## Related work, not novelty claims

- Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks*
  (Integrated Gradients, https://arxiv.org/abs/1703.01365).
- Dhamdhere, Sundararajan & Yan (2018), *How Important Is a Neuron?*
  (neuron conductance, https://arxiv.org/abs/1805.12233).
- Adebayo et al. (2018), *Sanity Checks for Saliency Maps*
  (https://arxiv.org/abs/1810.03292).

The present package combines existing attribution principles with a specific
scientific graphic grammar. A methodological novelty claim requires independent
validation against existing XAI and scientific visualization systems.
