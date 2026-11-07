# Symmetry alignment on independently fitted neural networks

**Research demonstration, not a claim that independently trained models are equivalent.**

## Key distinction

- **Exact gauge symmetry:** permuting hidden units and applying consistent sign flips to tanh neurons leaves the computed function unchanged for all inputs. This is an algebraic identity.
- **Empirical matching:** for two *independently fitted* models with equal architecture, ModelAtlas matches each hidden unit to a unit in the other model by minimizing a sign-invariant mean-squared activation discrepancy over **training data only**. Hungarian assignment gives a one-to-one pairing. The empirical pairing is not guaranteed unique or scientifically interpretable when units are highly correlated, inactive or duplicated.
- **Residual disagreement:** after matching, residual differences in held-out activations and predictive probabilities remain. We report these separately. Matching never changes the probability output of either fitted model.

## Equations

For a tanh network with row-vector inputs,

\[h_1=\tanh(xW_0+b_0),\quad h_2=\tanh(h_1W_1+b_1),\quad f(x)=\sigma(h_2W_2+b_2).\]

At each hidden layer, choose permutation \(P\) and signs \(S=\mathrm{diag}(\pm1)\). Reorder and sign-flip its incoming columns and biases, while applying corresponding row operations to the outgoing weights. Because \(\tanh(-u)=-\tanh(u)\), the function is unchanged.

Matching for independently trained networks minimizes signed activation trace discrepancy:

\[\min_{\pi,s_i\in\{-1,1\}}\sum_{i=1}^m\frac{1}{n}\sum_{j=1}^{n}(h^{A}_i(x_j)-s_i h^B_{\pi(i)}(x_j))^2.\]

This matching objective is **not** a functional equivalence test. It is a coordinate convention allowing meaningful visual comparison of internal units.

## Real-data experiments

Run `python examples/real_world_symmetry.py` after `pip install -e '.[examples,test]'`. The code uses only datasets included with scikit-learn, and writes a metrics JSON and PNG/SVG/PDF figures for each experiment. No network calls, cached models, or hidden private datasets are needed.

- **Wine (178 samples):** binary classification of wine class 0 versus the other classes using the first eight measured chemical properties.
- **Wisconsin Diagnostic Breast Cancer (569 samples):** benign/malignant classification from eight selected real measurements. This is a visualization benchmark, not medical advice or a clinical diagnostic tool.

For both experiments: a stratified 69/31 train/test split, common training-fitted scaler, two independently trained tanh MLPClassifier models (16 and 9 hidden units) with seeds 4 and 29, correspondences calculated on training observations only. Holdout data are used only to report predictive differences and residual activations. The displayed observation is selected as the held-out case of largest model disagreement **for illustration**, not as a representative example.

### What each figure means

- `independent_networks_aligned`: same input and layout for models A and B; a tile's color is its tanh activation for one observed input; thin links are signed fitted weights, not attributions. Matching only reorders/sign-flips B's neuron coordinates. Three circled second-layer neurons have largest absolute logit terms \(|h_{2,j}W_{2,j}|\) in the displayed observation, **not** causal importance.
- `residual_atlas`: RMS difference in matched hidden activation on held-out observations; edges encode signed differences between **aligned fitted weights**. This is not a canonical invariant metric, because tied/degenerate matches can change the displayed residuals.
- `matching_diagnostic`: mean matched-neuron correlations before/after alignment, plus an **unaligned in prediction space** scatterplot of independently trained probabilities. Any difference between model probabilities is unaffected by alignment.

The oblique 2.5D drawing is a deterministic coordinate layout, not a discovered intrinsic neural manifold. Do not interpret distances between tiles as distances in feature or function space.

## Limitations and next experiments

This renderer supports two-hidden-layer 8–16–9–1 networks; the alignment core supports other equal-width tanh architectures. ReLU positive-rescaling symmetries, convolutional channels, transformer heads, arbitrary latent basis changes, and different hidden widths are not addressed. Test confidence intervals across many training seeds, held-out samples from shifted populations, and identification-aware photo-z datasets before claiming scientific benefit.

## References

Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*, ICLR (2023); Kornblith et al., *Similarity of Neural Network Representations Revisited*, ICML (2019).
