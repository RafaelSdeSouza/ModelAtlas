"""Real astronomical example: NOAA yearly sunspot numbers, 1700--2008.

Reproducible offline: bundled public-domain dataset originated from the
National Geophysical Data Center, exposed in statsmodels.datasets.sunspots.
No synthetic observations are generated. SOM training is stochastic, seeded.

Run:
    python examples/solar_activity.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from modelatlas import SOMMap, train_som, visualize, plot_metric, plot_predictions

ROOT=Path(__file__).resolve().parents[1]


raw=pd.read_csv(ROOT/'data/sunspots_1700_2008.csv')
Y=raw['YEAR'].to_numpy(dtype=int)
S=raw['SUNACTIVITY'].to_numpy(dtype=float)
# Five real observations per overlapping 9-year profile.
offsets=np.array([-4,-2,0,2,4],dtype=int)
center=np.arange(4,len(S)-4)
Xraw=S[center[:,None]+offsets]
mu=Xraw.mean() ; sd=Xraw.std()
X=(Xraw-mu)/sd
weights=train_som(X,grid=(8,8),seed=47,steps=12000)
M=SOMMap(weights,samples=X,
         feature_labels=('Sₜ₋₄','Sₜ₋₂','Sₜ','Sₜ₊₂','Sₜ₊₄'),
         feature_units=('sunspots',)*5,
         display_codebook=weights*sd+mu)

# Mark the best-matching unit of the year of maximum solar activity.
obs=np.argmax(Xraw[:,2]); chosen_year=int(Y[center[obs]])
bmu=tuple(int(v) for v in M.best_matching_units(X[obs:obs+1])[0])
print('Map:', M.rows,'x',M.cols)
print('Year highlighted:',chosen_year,'activity',Xraw[obs,2], 'BMU',bmu)
print('BMU prototype in original sunspot units:',np.round(M.display_codebook[bmu],1))
metrics=M.diagnostics();print('SOM diagnostics',metrics)

out=ROOT/'outputs';out.mkdir(exist_ok=True)
fig,ax=visualize(M,color_by=2,labels=('Sₜ₋₄','Sₜ₋₂','Sₜ','Sₜ₊₂','Sₜ₊₄'),
                 highlight=bmu,depth=.40,azimuth=1.0,figsize=(5.0,4.85))
for suffix in ['png','svg','pdf']:
    fig.savefig(out/f'sunspot_som_oblique.{suffix}',dpi=275,bbox_inches='tight',
                pad_inches=.015,facecolor='white')
plt.close(fig)
for metric,cmap in [('occupancy','Greys'),('u_matrix','magma'),(0,'viridis')]:
    fig,ax=plot_metric(M,metric,cmap=cmap,cell_labels=(metric=='occupancy'))
    name='sunspot_'+('early_component' if metric==0 else metric)
    for suffix in ['png','svg']:
        fig.savefig(out/f'{name}.{suffix}',dpi=230,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

# Inspect representative learned *profiles*. Use locations picked entirely
# from fitted prototypes (not arbitrary hand-drawn curves).
w=M.display_codebook
flat=w.reshape(-1,5)
labels=[('Low at t',np.argmin(flat[:,2])),('High at t',np.argmax(flat[:,2])),
        ('Rising',np.argmax(flat[:,-1]-flat[:,0])),
        ('Declining',np.argmin(flat[:,-1]-flat[:,0]))]
fig,ax=plt.subplots(figsize=(4.45,2.9),dpi=220)
cols=['#3054A0','#D04633','#248B74','#8A62AA']
for (name,ind),color in zip(labels,cols):
    ax.plot(offsets,flat[ind],color=color,lw=2,marker='o',markersize=3.7,
            label=name)
ax.set_xlabel('Year relative to profile center',fontsize=9,color='#303D4E')
ax.set_ylabel('Yearly sunspot number',fontsize=9,color='#303D4E')
ax.set_xticks(offsets)
ax.tick_params(length=0,labelsize=8)
ax.grid(color='#77849B',alpha=.16,lw=.65)
for spine in ax.spines.values():spine.set_visible(False)
ax.legend(frameon=False,fontsize=8,loc='upper left',ncol=2)
fig.tight_layout(pad=.8)
for suffix in ['png','svg']:
    fig.savefig(out/f'sunspot_prototypes.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.045)
plt.close(fig)

# Generalization 1: linear autoregression. Different visual grammar: actual
# fitted coefficient signs rather than a fabricated SOM topology.
base=np.arange(4,len(S)-1)
Xfuture=np.column_stack([S[base-j] for j in range(4,-1,-1)])
yfuture=S[base+1]
train=Y[base]<=1920
test=Y[base]>=1930
mm=Xfuture[train].mean(axis=0); ss=Xfuture[train].std(axis=0)
ss=np.maximum(ss,1e-8)
Xfit=(Xfuture-mm)/ss
ridge=Ridge(alpha=8.).fit(Xfit[train],yfuture[train])
print('Ridge held-out chronological R2:',r2_score(yfuture[test],ridge.predict(Xfit[test])))
print('Persistence baseline R2:',r2_score(yfuture[test],Xfuture[test,-1]))
fig,_=visualize(ridge,kind='linear',labels=['t−4','t−3','t−2','t−1','t'],
                xlabel='sunspots per 1 SD increase in input')
for suffix in ['png','svg']:
    fig.savefig(out/f'sunspot_linear_regression.{suffix}',dpi=260,
                bbox_inches='tight',pad_inches=.03)
plt.close(fig)
fig,_=plot_predictions(yfuture[test],ridge.predict(Xfit[test]),
                       x=Y[base][test],baseline=Xfuture[test,-1],
                       xlabel='Year (chronological test set)',ylabel='Annual sunspot number')
for suffix in ['png','svg']:
    fig.savefig(out/f'sunspot_predictive_check.{suffix}',dpi=260,
                bbox_inches='tight',pad_inches=.03)
plt.close(fig)

# Generalization 2: small MLP, network arcs show fitted weights.
yscalef=np.std(yfuture[train]);ymean=np.mean(yfuture[train])
mlp=MLPRegressor(hidden_layer_sizes=(7,),activation='tanh',alpha=5.,
                 random_state=9,max_iter=1750,tol=1e-6,n_iter_no_change=150)
mlp.fit(Xfit[train],(yfuture[train]-ymean)/yscalef)
pred=mlp.predict(Xfit[test])*yscalef+ymean
print('MLP held-out chronological R2:',r2_score(yfuture[test],pred))
fig,_=visualize(mlp,kind='mlp',input_labels=['t−4','t−3','t−2','t−1','t'],
                output_labels=['t+1'],trim_quantile=.27)
for suffix in ['png','svg']:
    fig.savefig(out/f'sunspot_neural_network.{suffix}',dpi=260,
                bbox_inches='tight',pad_inches=.02)
plt.close(fig)

report={**metrics,'highlight_year':chosen_year,'highlight_bmu':bmu,
    'highlight_profile':np.round(M.display_codebook[bmu],2).tolist(),
    'input_mean':float(mu),'input_sd':float(sd),
    'n_years':int(len(Y)), 'n_profiles':int(len(X)),
    'lag_offsets':offsets.tolist(),
    'ridge_test_r2':float(r2_score(yfuture[test],ridge.predict(Xfit[test]))),
    'persistence_baseline_test_r2':float(r2_score(yfuture[test],Xfuture[test,-1])),
    'mlp_test_r2':float(r2_score(yfuture[test],pred))}
(out/'model_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print('Output directory:',out)
