"""
Fisher–Rao geometry and stable model neighborhoods
==================================================

Different fitted classifiers are compared as Bernoulli predictive distributions
on the same reference observations. This is model-family agnostic.
Connections denote 2-nearest-neighbor relationships stable under reference
sample bootstrap; they are not neural network weights or graph topology.
"""

import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
from modelatlas.behavior_geometry import build_probability_atlas
from modelatlas.behavior_plot import plot_probability_atlas

# Reuse the exact data split and fitted models from the functional-atlas demo.
# Standalone reference experiment; no coupling to another gallery script.
import copy
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from modelatlas.functional import model_predictions

data=load_breast_cancer()
col=[0,1,2,4,5,6,7,20]
X=data.data[:,col]
y=(data.target==0).astype(int)
A,B,y_train,y_ref=train_test_split(X,y,test_size=.46,random_state=1729,stratify=y)
scaler=StandardScaler().fit(A)
A=scaler.transform(A); B=scaler.transform(B)
models={
    'GLM':LogisticRegression(C=.9,max_iter=1500,random_state=1).fit(A,y_train),
    'Regularized GLM':LogisticRegression(C=.035,max_iter=1500,random_state=1).fit(A,y_train),
    'MLP A':MLPClassifier(hidden_layer_sizes=(12,7),activation='tanh',solver='lbfgs',
                           alpha=.25,max_iter=650,random_state=2).fit(A,y_train),
    'Random forest':RandomForestClassifier(n_estimators=130,max_depth=5,min_samples_leaf=6,
                                           random_state=4,n_jobs=1).fit(A,y_train),
    'Extra Trees':ExtraTreesClassifier(n_estimators=120,max_depth=5,min_samples_leaf=5,
                                       random_state=19,n_jobs=1).fit(A,y_train),
    'RBF SVM':SVC(kernel='rbf',C=2.3,probability=True,random_state=4).fit(A,y_train),
    'KNN':KNeighborsClassifier(n_neighbors=17).fit(A,y_train),
    'Gaussian NB':GaussianNB().fit(A,y_train),
}
permuted=copy.deepcopy(models['MLP A'])
rng=np.random.default_rng(12)
for layer in range(len(permuted.coefs_)-1):
    perm=rng.permutation(permuted.coefs_[layer].shape[1])
    permuted.coefs_[layer]=permuted.coefs_[layer][:,perm]
    permuted.intercepts_[layer]=permuted.intercepts_[layer][perm]
    permuted.coefs_[layer+1]=permuted.coefs_[layer+1][perm,:]
models['MLP A (permuted)']=permuted
predictions=model_predictions(models,B)
families={name:('MLP' if name.startswith('MLP') else
                'GLM' if 'GLM' in name else
                'Forest' if 'forest' in name or 'Trees' in name else
                'Kernel' if 'SVM' in name or 'KNN' in name else 'Bayes')
          for name in predictions}
atlas=build_probability_atlas(predictions, geometry='fisher_rao', n_bootstrap=250, seed=120)
labels={
    'GLM':'GLM','Regularized GLM':'rGLM','MLP A':'MLP',
    'Random forest':'RF','Extra Trees':'ET','RBF SVM':'SVM',
    'KNN':'kNN','Gaussian NB':'NB'
}
fig,_=plot_probability_atlas(atlas,family=families,label_map=labels)
output=os.environ.get('MODELATLAS_FIGURE_DIR')
if output:
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(dest/f'05_probability_geometry.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
    pairs=[]
    for i in range(len(atlas.unique_names)):
        for j in range(i+1,len(atlas.unique_names)):
            if atlas.edge_frequencies[i,j]>.5:
                pairs.append({'a':atlas.unique_names[i],'b':atlas.unique_names[j],
                              'edge_bootstrap_frequency':round(float(atlas.edge_frequencies[i,j]),3),
                              'fisher_rao_distance':round(float(atlas.distances[i,j]),4)})
    (dest/'05_probability_geometry.json').write_text(json.dumps({
        'dataset':'Wisconsin Diagnostic Breast Cancer (sklearn)',
        'model_parameterizations':len(atlas.names),
        'distinct_reference_prediction_profiles':len(atlas.unique_names),
        'reference_observations':atlas.n_reference,
        'metric':atlas.geometry,
        'retained_3d':atlas.retained_3d,
        '3d_stress':atlas.distance_stress,
        'bootstrap_replicates':atlas.n_bootstrap,
        'equivalence_classes':[list(x) for x in atlas.equivalence_classes],
        'stable_edges':pairs
    },indent=2))
if not output:
    plt.show()
