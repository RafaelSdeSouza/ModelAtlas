"""Reproduce ModelAtlas's XAI 2.5D neural-network demonstration on real wine data.

Run: python examples/wine_explanation_atlas.py
Uses eight measured chemical features, not simulated training inputs.
All transformations are fitted to the training split only.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings
import sys
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from modelatlas.explanation import explain_mlp
from modelatlas.explanation_plot import plot_explanation_atlas

FEATURES=[0,1,2,4,5,6,9,12]
SHORT_NAMES=('alcohol','malic acid','ash','magnesium','phenols','flavanoids','colour','proline')


def fit_example():
    dataset=load_wine()
    X=dataset.data[:,FEATURES]
    y=(dataset.target==0).astype(int)
    train,test=train_test_split(np.arange(len(y)),test_size=.32,stratify=y,random_state=31)
    scaler=StandardScaler().fit(X[train]);xt=scaler.transform(X[train]);xte=scaler.transform(X[test])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=UserWarning)
        model=MLPClassifier(hidden_layer_sizes=(16,9),activation='tanh',
            alpha=.16,learning_rate_init=.003,max_iter=900,random_state=21,
            n_iter_no_change=80)
        model.fit(xt,y[train])
    return model,xt,xte,y[test],train,test


def main(out=None):
    out=Path(out) if out else Path(__file__).resolve().parents[1]/'docs'/'figures'
    out.mkdir(parents=True,exist_ok=True)
    model,xt,xte,yte,train,test=fit_example()
    scores=model.predict_proba(xte)[:,1]
    auc=float(roc_auc_score(yte,scores))
    choices={
        'class_one':int(np.argmin(np.abs(scores-.85))),
        'class_zero':int(np.argmin(np.abs(scores-.007))),
    }
    metadata={'dataset':'UCI Wine via sklearn.load_wine',
              'features':SHORT_NAMES,'train_size':len(xt), 'test_size':len(xte),
              'test_auc':auc,'seed_split':31,'seed_model':21,
              'baseline':'training means in standardized coordinates',
              'target':'class-0 logit (wine class 0 versus classes 1 and 2)',
              'quadrature_steps':256,'samples':{}}
    for key,idx in choices.items():
        exp=explain_mlp(model,xte[idx],baseline=np.zeros(8),steps=256)
        fig,_=plot_explanation_atlas(model,exp,X_reference=xt,feature_names=SHORT_NAMES)
        for suffix in ('png','pdf','svg'):
            fig.savefig(out/f'wine_{key}_explanation_atlas.{suffix}',dpi=250,
                        facecolor='white',bbox_inches='tight',pad_inches=.075)
        import matplotlib.pyplot as plt
        plt.close(fig)
        metadata['samples'][key]={
            'test_index':int(idx),'original_index':int(test[idx]),'true_class':int(yte[idx]),
            'probability':exp.probability,'baseline_probability':exp.baseline_probability,
            'logit_change':exp.logit_change,'input_IG':exp.node_contributions[0].tolist(),
            'max_layer_conservation_residual':exp.max_layer_residual,
            'max_neuron_conservation_residual':exp.max_node_residual,
        }
    (out/'wine_explanation_metrics.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
    return metadata

if __name__=='__main__':main()
