"""Real-data check of symmetry-aware matching on independent MLPs.

Run from editable install: python examples/real_world_symmetry.py
This script does not use test observations in training the models or matching.
"""
from pathlib import Path
import json
import warnings

import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from modelatlas.symmetry import match_tanh_mlp, activations, representational_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs'/'figures'/'symmetry'


def real_datasets():
    wine=load_wine()
    cancer=load_breast_cancer()
    series=np.genfromtxt(ROOT/'data'/'sunspots_1700_2008.csv', delimiter=',', skip_header=1)
    counts=series[:,1]
    # At year t, predict whether solar activity increases at year t+1 using
    # only the eight annual counts through year t. Chronological holdout.
    solar_X=np.stack([counts[t-7:t+1] for t in range(7,len(counts)-1)])
    solar_y=np.array([counts[t+1]>counts[t] for t in range(7,len(counts)-1)], dtype=int)
    return {
        'solar': dict(X=solar_X,y=solar_y,feature_names=tuple(f'activity t-{7-j}' for j in range(8)),
                      title='Solar sunspot activity',label='activity rises next year',chronological=True),
        'wine': dict(X=wine.data[:,:8],y=(wine.target==0).astype(int),
                     feature_names=tuple(wine.feature_names[:8]),
                     title='Wine chemistry',label='class 0 wine'),
        'cancer': dict(X=cancer.data[:,[0,1,2,3,6,7,20,27]],
                       y=(cancer.target==0).astype(int),
                       feature_names=tuple(cancer.feature_names[i] for i in [0,1,2,3,6,7,20,27]),
                       title='Breast-cancer measurements',label='malignant'),
    }


def train_case(name,seed1=4,seed2=29):
    from modelatlas.symmetry_plot import plot_independent_pair, plot_alignment_diagnostic, plot_residual_atlas
    d=real_datasets()[name];X=d['X'];y=d['y']
    if d.get('chronological',False):
        # Do not shuffle overlapping sunspot windows into training and test.
        cut=int(.72*len(y))
        Xtr,Xte,ytr,yte=X[:cut],X[cut:],y[:cut],y[cut:]
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.31,random_state=2026,stratify=y)
    standardizer=StandardScaler().fit(Xtr)
    train=standardizer.transform(Xtr)
    test=standardizer.transform(Xte)
    models=[]
    for seed in (seed1,seed2):
        m=MLPClassifier(hidden_layer_sizes=(16,9),activation='tanh',solver='adam',
                        alpha=.12,max_iter=950,learning_rate_init=.003,random_state=seed,
                        tol=1e-6,n_iter_no_change=40)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.fit(train,ytr)
        models.append(m)
    A,B=models
    align=match_tanh_mlp(A,B,train,X_check=test)
    _,_,pA=activations(A.coefs_,A.intercepts_,test)
    _,_,pB=activations(B.coefs_,B.intercepts_,test)
    metric=representational_metrics(A,align,test)
    ix=int(np.argmax(np.abs(pA-pB)))
    report={
        'dataset':name,'n_train':int(train.shape[0]),'n_test':int(test.shape[0]),
        'n_features':int(X.shape[1]),'selected_test_index':ix,
        'split':'chronological' if d.get('chronological',False) else 'stratified random',
        'seed_A':seed1,'seed_B':seed2,
        'test_auc_A':float(roc_auc_score(yte,pA)),
        'test_auc_B':float(roc_auc_score(yte,pB)),
        'test_accuracy_A':float(accuracy_score(yte,pA>=.5)),
        'test_accuracy_B':float(accuracy_score(yte,pB>=.5)),
        'raw_mean_corr_train':list(align.raw_mean_corr),
        'aligned_mean_corr_train':list(align.aligned_mean_corr),
        'raw_mean_abs_train':list(align.raw_mean_abs),
        'aligned_mean_abs_train':list(align.aligned_mean_abs),
        'aligned_metrics_test':metric,
        'selected_p_A':float(pA[ix]),'selected_p_B':float(pB[ix]),
        'invariance_error':align.prediction_preservation_error,
    }
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    plot_independent_pair(A,align,test[ix],d['title'],d['label'],report,out)
    plot_residual_atlas(A,align,test,d['title'],report,out)
    plot_alignment_diagnostic(A,B,align,train,test,yte,d['title'],report,out)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report


if __name__=='__main__':
    for name in real_datasets():
        metric=train_case(name)
        print(json.dumps(metric,indent=2))