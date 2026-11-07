"""Reproduce the true 5-input-node / 81-output-neuron SOM architecture.

Uses the already fitted prototype matrix saved by photoz_identification.py
and the public SYNTHETIC catalogue from v0.2. Does not re-fit or use hidden z.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelatlas import SOMMap, PhotoZCells, plot_photoz_architecture, plot_som_architecture

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'photoz'
arr=np.load(OUT/'fitted_som_codebook.npz')
W=arr['codebook']; mu=arr['feature_mean']; sd=arr['feature_std']
cat=pd.read_csv(OUT/'mock_parent_photometry.csv')
labels=('g−r','r−i','i−z','z−y','i')
X=cat[['g_r','r_i','i_z','z_y','i_mag']].to_numpy()
Xscaled=(X-mu)/sd
som=SOMMap(W,samples=Xscaled,feature_labels=labels,
           display_codebook=W*sd[None,None,:]+mu[None,None,:])
zspec=cat['spectroscopic_z'].to_numpy(dtype=float)
B=PhotoZCells(som,Xscaled,zspec,attempted=cat['attempted'].to_numpy(dtype=bool)).identify(z0=1.2,gamma=.14)
ranked=B.rank_cells(top=1)[0]; highlight=(ranked['row'], ranked['col'])

# The missing original architecture view: real nodes and links, colors are
# the photo-z identified-width metric already derived in the prior example.
fig,ax=plot_photoz_architecture(B,som,metric='identified_width',
                show_connections='all',emphasize_highlight=True,
                highlight=highlight, cmap='turbo',figsize=(5.55,4.9))
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'photoz_som_architecture.{ext}',dpi=300,
               bbox_inches='tight',pad_inches=.015, transparent=(ext!='png'))
print('Arch metadata',ax._modelatlas_architecture)
plt.close(fig)

# Same 81 prototypes and same input links, but colors show a real learned
# prototype coordinate, providing a direct model-architecture interpretation.
fig,ax=plot_som_architecture(som,metric_label='g−r · prototype value',
                show_connections='all',highlight=highlight,
                figsize=(5.55,4.9),cmap='turbo')
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'photoz_som_codebook_network.{ext}',dpi=300,
               bbox_inches='tight',pad_inches=.015,transparent=(ext!='png'))
plt.close(fig)
