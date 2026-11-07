"""
Functional geometry and information lattices
===========================================

Different fitted model families are positioned by their **predictions** on
common held-out observations, not by incompatible internal parameters. A
hidden-neuron permutation produces a second parameterization of exactly the
same neural network function. The information lattice then analyzes how much
of that model's prediction behavior remains unresolved when only a specified
set of binned input features is observed.

The dataset is the public Wisconsin Diagnostic Breast Cancer collection from
scikit-learn. Classification accuracy is not the aim of the demonstration.
The finite reference sample is an illustration, not a proof of model equality
outside those observations. The lattice shows prediction variation, not
confidence intervals or partial identification.
"""

import copy
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modelatlas.functional import (functional_map, information_lattice,
                                  find_empirical_equivalences, model_predictions,
                                  monotonicity_violations)
from modelatlas.functional_plot import (plot_functional_atlas,
                                        plot_information_lattice,
                                        plot_prediction_fibers)


def build():
    data=load_breast_cancer()
    cols=[0,1,2,4,5,6,7,20]
    X=data.data[:,cols]
    y=(data.target==0).astype(int)  # probability of malignant class
    X_train, X_ref, y_train, y_ref = train_test_split(X,y,test_size=.46,random_state=1729,stratify=y)
    scaler=StandardScaler().fit(X_train)
    A=scaler.transform(X_train)
    B=scaler.transform(X_ref)
    models={
        'GLM': LogisticRegression(C=.9,max_iter=1500,random_state=1).fit(A,y_train),
        'Regularized GLM':LogisticRegression(C=.035,max_iter=1500,random_state=1).fit(A,y_train),
        'MLP A':MLPClassifier(hidden_layer_sizes=(12,7),activation='tanh',solver='lbfgs',alpha=.25,max_iter=650,random_state=2).fit(A,y_train),
        'Random forest':RandomForestClassifier(n_estimators=130,max_depth=5,min_samples_leaf=6,random_state=4,n_jobs=1).fit(A,y_train),
        'Extra Trees':ExtraTreesClassifier(n_estimators=120,max_depth=5,min_samples_leaf=5,random_state=19,n_jobs=1).fit(A,y_train),
        'RBF SVM':SVC(kernel='rbf',C=2.3,probability=True,random_state=4).fit(A,y_train),
        'KNN':KNeighborsClassifier(n_neighbors=17).fit(A,y_train),
        'Gaussian NB':GaussianNB().fit(A,y_train),
    }
    # Demonstrate an *exact* hidden-unit permutation orbit within the MLP family.
    # It is crucial to permute output connections together with hidden units.
    clone=copy.deepcopy(models['MLP A'])
    rng=np.random.default_rng(12)
    for layer in range(len(clone.coefs_)-1):
        perm=rng.permutation(clone.coefs_[layer].shape[1])
        clone.coefs_[layer]=clone.coefs_[layer][:,perm]
        clone.intercepts_[layer]=clone.intercepts_[layer][perm]
        clone.coefs_[layer+1]=clone.coefs_[layer+1][perm,:]
    models['MLP A (permuted)']=clone
    pred=model_predictions(models,B)
    fm=functional_map(pred)
    families={name:('MLP' if name.startswith('MLP') else 'GLM' if 'GLM' in name else
                    'Forest' if 'forest' in name or 'Trees' in name else
                    'Kernel' if 'SVM' in name or 'KNN' in name else 'Bayes') for name in pred}
    brier={name:brier_score_loss(y_ref,p) for name,p in pred.items()}
    auc={name:roc_auc_score(y_ref,p) for name,p in pred.items()}
    # Thresholds are determined by training data, then applied to fixed held-out sample.
    pick=[0,1,3,5]
    thresholds=np.median(A[:,pick],axis=0)
    # X_ref was standardized using training moments. B[:, pick] is comparable.
    lattice=information_lattice(B,pred['MLP A'],feature_indices=pick,thresholds=thresholds,feature_names=('R','T','S','C'))
    return B,y_ref,models,pred,fm,lattice,families,brier,auc


if __name__=='__main__' or __name__.startswith('sphinx_gallery'):
    B,y,models,pred,fm,lattice,families,brier,auc=build()
    output_dir = os.environ.get('MODELATLAS_FIGURE_DIR')
    out = Path(output_dir) if output_dir else None
    if out is not None: out.mkdir(parents=True,exist_ok=True)
    fig,ax=plot_functional_atlas(fm,family=families,quality=brier)
    if out is not None:
        for ext in ('png','svg','pdf'):
            fig.savefig(out/f'01_functional_atlas.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.show()
    fig,ax=plot_information_lattice(lattice)
    if out is not None:
        for ext in ('png','svg','pdf'):
            fig.savefig(out/f'02_information_lattice.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.show()
    fig,ax=plot_prediction_fibers(B,pred['MLP A'],features=(0,1))
    if out is not None:
        for ext in ('png','svg','pdf'):
            fig.savefig(out/f'03_projection_fibers.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    plt.show()
    d=fm.distances
    pi=fm.names.index('MLP A');pj=fm.names.index('MLP A (permuted)')
    stats={
        'n_reference':int(len(B)), 'num_models':len(models),
        'functional_MDS_2D_variance_fraction':fm.variance_retained,
        'functional_distance_stress':fm.stress,
        'functional_MDS_3D_variance_fraction':fm.variance_retained_3d,
        'functional_distance_stress_3D':fm.stress3,
        'exact_hidden_permutation_max_pred_diff':float(np.max(np.abs(pred['MLP A']-pred['MLP A (permuted)']))),
        'functional_distance_under_permutation':float(d[pi,pj]),
        'empirical_equivalences':[[fm.names[i] for i in group] for group in find_empirical_equivalences(fm)],
        'monotonicity_violations':len(monotonicity_violations(lattice)),
        'baseline_pred_variance':lattice.baseline_variance,
        'variance_after_4_binned_features':lattice.nodes[-1].residual_variance,
        'shapley':dict(zip(lattice.feature_names,map(float,lattice.shapley_variance_reduction))),
        'brier':brier,'auc':auc,
        'mean_absolute_prob_diff_glm_mlp':float(np.mean(np.abs(pred['GLM']-pred['MLP A']))),
    }
    if out is not None: (out/'diagnostics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2))
