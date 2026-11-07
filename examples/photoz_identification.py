"""Reproducible photo-z population-inference demonstration with a synthetic SED mock.

No HSC, DESI, LSST or observed galaxy data are claimed. Synthetic ugrizy-like
passbands deliberately create photometric degeneracy. SOM is fitted ONLY to the
parent photometry; missing redshifts are not used for any reported bound.

Run:  python examples/photoz_identification.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import wasserstein_distance
from modelatlas import (SOMMap,train_som,PhotoZCells,plot_photoz_map,plot_photoz_diagnostics)

OUT=Path(__file__).resolve().parents[1]/'outputs'/'photoz'
OUT.mkdir(exist_ok=True,parents=True)
rng=np.random.default_rng(20261106)
N=5200

# Three broad *mock* population components: low-z, intermediate-z, high-z.
component=rng.choice(3,size=N,p=[.52,.30,.18])
z=np.empty(N)
z[component==0]=.06+rng.gamma(2.0,.27,size=np.sum(component==0))
z[component==1]=rng.normal(1.10,.22,size=np.sum(component==1))
z[component==2]=rng.normal(1.83,.24,size=np.sum(component==2))
z=np.clip(z,.05,2.47)
old=rng.random(N)<(.28+.32*(z<.8))
D4000=np.where(old,rng.normal(.95,.15,N),rng.normal(.25,.09,N))
D4000=np.clip(D4000,.08,1.35)
Av=np.where(old,.18,.38)+rng.gamma(1.35,.13,size=N)
alpha=np.where(old,-.85,.55)+rng.normal(0,.20,N)

# Integrate an illustrative SED in five broad Gaussian optical filters.
# This is a toy physical sketch (4000 Å break, dust, continuum and lines),
# not a calibrated instrumental response nor a real survey catalogue.
wavelength=np.array([4760,6220,7630,9050,10030.])
sigma=np.array([510,600,620,640,650.])
grid=np.linspace(-2.3,2.3,17)
weights=np.exp(-.5*grid**2);weights/=weights.sum()
lam_obs=wavelength[None,:,None] + sigma[None,:,None]*grid[None,None,:]
lam_rest=lam_obs/(1+z[:,None,None])
continuum=(lam_rest/5500.)**(-alpha[:,None,None])
continuum*=np.where(lam_rest<4000,1/(1+D4000[:,None,None]),1.)
continuum*=np.exp(-Av[:,None,None]*(lam_rest/5500.)**(-.82))
for center,strength in [(3727,.75),(5007,.42),(6563,.55)]:
    amp=np.where(old,.09,1.)[:,None,None]*strength
    continuum*=1+amp*np.exp(-.5*((lam_rest-center)/45)**2)
flux=np.sum(continuum*weights[None,None,:],axis=2)
mag=-2.5*np.log10(np.maximum(flux,1e-12))
mag-=mag[:,[2]] # photometric colors relative to i
im=20.8+1.10*z+.34*old+rng.normal(0,.67,N)
im=np.clip(im,19.2,25.5)
mag+=im[:,None]
# Magnitude uncertainty becomes larger for faint galaxies.
noise_sd=np.clip(.015+.017*10**(.24*(im-21.6)),.015,.28)
mag+=rng.normal(0,noise_sd[:,None],(N,5))
X=np.column_stack([mag[:,0]-mag[:,1],mag[:,1]-mag[:,2],
                   mag[:,2]-mag[:,3],mag[:,3]-mag[:,4],mag[:,2]])
labels=('g−r','r−i','i−z','z−y','i')
# Each color has its own reasonable scale, while i magnitude isn't allowed
# to dominate a Euclidean distance solely because of units.
mu=X.mean(axis=0);sd=X.std(axis=0)
X_scaled=(X-mu)/sd

# Spectroscopy is targeted non-randomly in magnitude/photometric space;
# successful measurement also depends on latent z (spectral-feature visibility).
def logistic(v):return 1/(1+np.exp(-v))
p_attempt=(.18+.78*logistic(1.45*(23.15-im))) * (0.90+.10*(X[:,0]<1.1))
p_attempt=np.clip(p_attempt,.02,.96)
attempted=rng.random(N)<p_attempt
z_desert=np.exp(-.5*((z-1.43)/.28)**2)
p_succ=logistic(1.45*(23.45-im))*(.90-.53*z_desert)*np.where(old,.78,1.)
success=attempted & (rng.random(N)<p_succ)
zspec=np.where(success,z,np.nan)

# Fit a SOM using only target photometry. No redshift enters training.
w=train_som(X_scaled,grid=(9,9),steps=8500,seed=41,learning_rate=.38)
som=SOMMap(w,samples=X_scaled,feature_labels=labels,
           display_codebook=w*sd[None,None,:]+mu[None,None,:])
catalogue=PhotoZCells(som,X_scaled,zspec,attempted=attempted)
z0=1.20
Gamma=.14   # conditional W1 sensitivity radius, in redshift units
B=catalogue.identify(z0=z0,gamma=Gamma)
B0=catalogue.identify(z0=z0,gamma=0)
Bwide=catalogue.identify(z0=z0,gamma=.35)
Bfree=catalogue.identify(z0=z0,gamma=None)

# Highlight the cell contributing most to CURRENT population bound width.
# This is not the expected utility of taking another spectrum. The selection
# uses only observed spectroscopy, target photometry and stated assumptions.
ranked=B.rank_cells(top=8)
highlight=(ranked[0]['row'],ranked[0]['col'])
print('Highlight',highlight, 'N=',B.n_target[highlight], 'spec=',B.n_spec[highlight],
      'bounds=',B.lower[highlight],B.upper[highlight])

metrics=['target_occupancy','spectroscopic_coverage','identified_width',
         'observed_high_fraction','attempt_fraction','success_given_attempt',
         'population_width_contribution']
for metric in metrics:
    fig,ax=plot_photoz_map(B,metric=metric,highlight=highlight,
                            figsize=(4.30,4.36))
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'photoz_{metric}.{ext}',dpi=260,transparent=(ext!='png'),
                    bbox_inches='tight',pad_inches=.015)
    plt.close(fig)

# Optional relief rendering: both hue and prism lift encode the scalar width.
fig,ax=plot_photoz_map(B,metric='identified_width',highlight=highlight,
                       elevation='metric',figsize=(4.30,4.36))
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'photoz_identified_width_relief.{ext}',dpi=260,
                bbox_inches='tight',pad_inches=.015)
plt.close(fig)

# Four interpretable cell layers, sharing exactly the same neuron geometry.
fig,axs=plot_photoz_diagnostics(B,highlight=highlight)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'photoz_four_diagnostics.{ext}',dpi=240,
                bbox_inches='tight',pad_inches=.025)
plt.close(fig)

# Independent, honest 1D summary: changing an assumption changes the
# identified *set*, not a statistical confidence interval.
fig,ax=plt.subplots(figsize=(6.6,3.45),dpi=220)
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=.24,right=.96,bottom=.23,top=.96)
variants=[('Exact transfer',B0),('W₁ ≤ 0.14',B),('W₁ ≤ 0.35',Bwide),('Unrestricted',Bfree)]
for i,(name,obj) in enumerate(variants):
    l,u=obj.population_interval();yy=3-i
    ax.plot([l,u],[yy,yy],color='#344962',lw=3.4,solid_capstyle='round',zorder=3)
    ax.scatter([l,u],[yy,yy],s=33,color='#344962',zorder=4)
    ax.text(u+.012,yy,f'[{l:.2f}, {u:.2f}]',fontsize=8.8,color='#3E5067',va='center')
theta=float(np.mean(z>z0))
observed=float(np.mean(z[success]>z0))
ax.axvline(theta,color='#D85F39',lw=1.6,ls='-',label='Mock target truth')
ax.axvline(observed,color='#4682A5',lw=1.4,ls='--',label='Usable-spectra fraction')
ax.set_yticks(np.arange(3,-1,-1),[label for label,_ in variants])
ax.set_xlim(0,1.06);ax.set_ylim(-.65,3.65)
ax.set_xlabel('Target fraction with z > 1.2',fontsize=10,color='#35435A')
ax.tick_params(axis='both',labelsize=9,length=0,pad=9)
ax.grid(axis='x',color='#BEC7D4',lw=.7,alpha=.30)
ax.legend(loc='upper right',frameon=False,fontsize=8.5)
for sp in ax.spines.values():sp.set_visible(False)
for ext in ('png','svg','pdf'):
    fig.savefig(OUT/f'identified_population_interval.{ext}',dpi=270,
                bbox_inches='tight',pad_inches=.08)
plt.close(fig)

# Genuine mock validation using hidden redshifts, only for the example's
# assumption audit. In real data, P_missing is not observed.
ids=catalogue.flat_cells
truth_in_bound=np.zeros_like(B.n_target,dtype=bool)
assumption_true=np.zeros_like(B.n_target,dtype=bool)
weighted_count=0;weighted_assumption=0;weighted_truth=0
w1values=[]
for k in np.unique(ids):
    mask=(ids==k);i,j=np.unravel_index(k,B.n_target.shape)
    if B.n_spec[i,j] and np.any(mask & ~success):
        distance=float(wasserstein_distance(z[mask & success],z[mask & ~success]))
        w1values.append(distance)
        assumption_true[i,j] = distance <= Gamma+1e-12
        weighted_count+=mask.sum();weighted_assumption+=int(assumption_true[i,j])*mask.sum()
    actual=float(np.mean(z[mask]>z0))
    truth_in_bound[i,j]=B.lower[i,j]-1e-9<=actual<=B.upper[i,j]+1e-9
    weighted_truth+=int(truth_in_bound[i,j])*mask.sum()

report={
    'catalogue': 'SYNTHETIC five-band SED mock; NOT real DESI/HSC/LSST data',
    'seed':20261106,'n_target':N,'features':list(labels),
    'source_photometry': 'toy SED with 4000A break, dust, lines, Gaussian passbands',
    'SOM_grid':[9,9],'SOM_steps':8500,'z_threshold':z0,
    'Gamma_assumption':'per-cell W1(P_missing(z), empirical P_usable(z)) <= Gamma',
    'Gamma':Gamma,
    'bounds':{x.replace(' ','_'):b.summary() for x,b in variants},
    'highlight_cell':[int(v) for v in highlight],
    'current_bound_width_contribution_top_cells':ranked,
    'highlight_N':int(B.n_target[highlight]),
    'highlight_spec':int(B.n_spec[highlight]),
    'highlight_interval':[float(B.lower[highlight]),float(B.upper[highlight])],
    'mock_target_truth':theta,
    'mock_usable_spectra_fraction':observed,
    'mock_target_fraction_inside_cellwise_bounds_weighted':weighted_truth/N,
    'mock_assumption_satisfied_target_weighted_among_cells_with_missing_and_spec':weighted_assumption/max(1,weighted_count),
    'mock_true_W1_cells_median':float(np.median(w1values)),
    'mock_true_W1_cells_p90':float(np.quantile(w1values,.9)),
    'SOM_diagnostics':som.diagnostics(),
}
(OUT/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
# Keep observed catalogue and truth separate, so absence of labels is obvious.
parent=pd.DataFrame(X,columns=['g_r','r_i','i_z','z_y','i_mag'])
parent['attempted']=attempted;parent['usable_spectrum']=success
parent['spectroscopic_z']=zspec
parent.to_csv(OUT/'mock_parent_photometry.csv',index=False,float_format='%.5f')
pd.DataFrame({'mock_true_z_NOT_OBSERVED':z}).to_csv(OUT/'mock_truth_validation_only.csv',index=False,float_format='%.5f')
np.savez_compressed(OUT/'fitted_som_codebook.npz',codebook=w,feature_mean=mu,feature_std=sd)
print(json.dumps(report,indent=2))
