"""
Different regression models, one functional geometry
====================================================

ModelAtlas can compare regressors with no knowledge of their internal
parameters. Coordinates come from RMS differences between predictions on one
held-out reference sample. Positions in this display are not parameter or
feature embeddings. The dataset is scikit-learn's diabetes regression set.
"""

from pathlib import Path
import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from modelatlas.functional import model_predictions, functional_map
from modelatlas.functional_plot import plot_functional_atlas

X,y=load_diabetes(return_X_y=True)
A,B,ya,yb=train_test_split(X,y,test_size=.35,random_state=127)
models={
 'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=12.)).fit(A,ya),
 'Random forest R':RandomForestRegressor(n_estimators=140,min_samples_leaf=6,random_state=1).fit(A,ya),
 'SVR':make_pipeline(StandardScaler(),SVR(C=16,epsilon=12)).fit(A,ya),
 'kNN R':make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=15)).fit(A,ya),
 'Boosted R':HistGradientBoostingRegressor(max_iter=110,random_state=1,max_leaf_nodes=5).fit(A,ya),
 'Tree R':DecisionTreeRegressor(max_depth=3,min_samples_leaf=14,random_state=1).fit(A,ya),
 'Mean':DummyRegressor().fit(A,ya),
}
pred=model_predictions(models,B)
geo=functional_map(pred)
family={'Ridge':'GLM','Random forest R':'Forest','SVR':'Kernel','kNN R':'Kernel',
 'Boosted R':'Forest','Tree R':'Forest','Mean':'Bayes'}
fig,ax=plot_functional_atlas(geo,family=family,figsize=(7.5,5.8))
out=os.environ.get('MODELATLAS_FIGURE_DIR')
if out:
    Path(out).mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg','pdf'):
        fig.savefig(Path(out)/f'04_regression_functional_atlas.{ext}',dpi=260,bbox_inches='tight',facecolor='white')
plt.show()
print(f'Diabetes regressors: 2D retained fraction {geo.variance_retained:.3f}; stress {geo.stress:.3f}')
