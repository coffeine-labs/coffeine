import numpy as np
from meegpowreg.covariance_transformers import (
    Diag,
    LogDiag,
    ExpandFeatures,
    Riemann,
    RiemannSnp,
    NaiveVec)

from meegpowreg.spatial_filters import (
    ProjIdentitySpace,
    ProjCommonSpace,
    ProjLWSpace,
    ProjRandomSpace,
    ProjSPoCSpace)

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV


def make_pipelines(
        frequency_bands=('low', 'delta', 'theta', 'alpha', 'beta'),
        scale='auto',
        n_components='full',
        reg=1.e-05,
        pipeline='riemann',
        shrink=1,
        rank='full',  # for RiemannWass
        method='upper',  # for NaiveVec: 'upper' 'upperlog' 'logdiag+upper'
        expand_feautures=None,
        # if expand=True, learns the interaction effect w/ last binary column,
        expander_column=-1,
        preprocessor=None,
        estimator=None,
        ridge_alphas=np.logspace(-3, 5, 100)):
    
    estimator_ = None
    preprocessor_ = None

    if estimator is None:
        estimator_ = RidgeCV(alphas=ridge_alphas)
    if preprocessor_ is None:
        preprocessor_ = StandardScaler()

    expander = None   
    if expand_feautures is not None:
        expander = make_column_transformer(
            *[(ExpandFeatures(expand=True), [band, expander_column])
              for band in frequency_bands],
            ('drop', expander_column))
 
    def _get_projector_vectorizer(steps):
        steps_ = [k(**v) for k, v in steps]
        return [(make_pipeline(*steps_), band) for band in frequency_bands]
    
    steps = list()
    if pipeline == 'riemann':
        steps = [
            (ProjCommonSpace, dict(scale=scale,
                                   n_compo=n_components,
                                   reg=reg)),
            (Riemann, dict(metric='riemann'))]
    elif pipeline == 'lw_riemann':
        steps = [(ProjLWSpace, dict(shrink=shrink)),
                 (Riemann, dict(metric='riemann'))]
    elif pipeline == 'diag':
        steps = [(ProjIdentitySpace, dict()),
                 (Diag, dict())]
    elif pipeline == 'logdiag':
        steps = [(ProjIdentitySpace, dict()),
                 (LogDiag, dict())]
    elif pipeline == 'random':
        steps = [(ProjRandomSpace, dict(n_compo=n_components)),
                 (LogDiag, dict())]
    elif pipeline == 'naive':
        steps = [(ProjIdentitySpace, dict()),
                 (NaiveVec, dict(method=method))]
    elif pipeline == 'spoc':
        steps = [(ProjSPoCSpace, dict(n_compo=n_components, scale=scale,
                                      reg=reg,
                                      shrink=shrink)),
                 (LogDiag, dict())]
    elif pipeline == 'riemann_wasserstein':
        steps = [(ProjIdentitySpace, dict()),
                 (RiemannSnp, dict(rank=rank))]

    pipeline_steps = [
        make_column_transformer(
            *_get_projector_vectorizer(steps),
            remainder='passthrough'),
        preprocessor,
        estimator]

    if expander is not None:
        pipeline_steps.insert(1, expander)
    filter_bank = make_pipeline(*pipeline_steps)

    return filter_bank
