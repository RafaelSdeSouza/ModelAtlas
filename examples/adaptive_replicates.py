"""Repeated mock comparison; NO photo-z redshifts used to train representations."""
import csv
import json
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from modelatlas.adaptive import growing_neural_gas,train_gtm
from modelatlas.som import train_som

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('_adaptive_comparison',HERE/'adaptive_photoz_comparison.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
metrics=[]
for seed in (39,59,79,99,119):
    data=m.generate_mock(6000,seed)
    a,b=train_test_split(np.arange(6000),test_size=.25,random_state=seed)
    standard=StandardScaler().fit(data['X'][a]); x=standard.transform(data['X'])
    som=train_som(x[a],grid=(8,8),steps=7500,seed=seed).reshape(64,-1)
    gtm=train_gtm(x[a],grid=(8,8),n_iter=45,ridge=.005,seed=seed).prototypes
    graph=growing_neural_gas(x[a],max_nodes=64,steps=15500,insertion_interval=230,max_edge_age=10,seed=seed)
    for name,w,adj in [('SOM',som,m.adjacency_grid((8,8))),('GTM',gtm,m.adjacency_grid((8,8))),('GNG',graph.prototypes,graph.adjacency)]:
        assigned=np.argmin(pairwise_distances(x,w),axis=1)
        d=m.evaluate(name,w,assigned,data['z'],data['z_spec'],x[b],adj,gamma=.16)
        row={'seed':seed,'method':name,'QE':d['test_quantization_error'],
             'within_cell_z_SD':d['weighted_within_cell_z_sd'],
             'identified_width':d['identification_width'],
             'W1_violation_fraction':d['audit_fraction_W1_violation_among_comparable_nodes'],
             'edges':d['edges'],'TE':d['test_topographic_error'],
             'truth_contained':d['interval_contains_truth']}
        metrics.append(row)
    print(f'finished mock seed={seed}',flush=True)
out=HERE.parent/'docs'/'adaptive_replicates.csv'
with open(out,'w',newline='') as file:
    writer=csv.DictWriter(file,metrics[0].keys());writer.writeheader();writer.writerows(metrics)
summary={}
for name in ('SOM','GTM','GNG'):
    rows=[a for a in metrics if a['method']==name]
    summary[name]={k:{'mean':float(np.mean([r[k] for r in rows])),
                      'sd':float(np.std([r[k] for r in rows],ddof=1))}
                   for k in ('QE','within_cell_z_SD','identified_width','W1_violation_fraction','edges','TE')}
    summary[name]['truth_contained_runs']=int(sum(r['truth_contained'] for r in rows))
(HERE.parent/'docs'/'adaptive_replicates_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))