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


def make_pipelines(names, pipeline='riemann', projection_params=None,
                   vectorization_params=None, expand_feautures=None,
                   expander_column=None, preprocessor=None, estimator=None):
    # XXX do proper doc string
    
    # put defaults here for projection and vectorization step
    projection_defaults = {
        'riemann': dict(scale=1, n_compo='full', reg=1.e-05),
        'lw_riemann': dict(shrink=1),
        'diag': dict(),
        'log_diag': dict(),
        'random': dict(n_compo='full'),
        'naive': dict(),
        'spoc': dict(n_compo='full', scale='auto', reg=1.e-05, shrink=1),
        'riemann_wasserstein': dict()
    }

    vectorization_defaults = {
        'riemann': dict(metric='riemann'),
        'lw_riemann': dict(metric='riemann'),
        'diag': dict(),
        'log_diag': dict(),
        'random': dict(),
        'naive': dict(method='upper'),
        'spoc': dict(),
        'riemann_wasserstein': dict(rank='full')
    }

    # update defaults
    projection_params_ = projection_defaults[pipeline]
    if projection_params is not None:
        projection_params_.update(**projection_params)
        
    vectorization_params_ = vectorization_defaults[pipeline]
    if vectorization_params is not None:
        vectorization_params_.update(**vectorization_params)

    # XXX not yet done here
    expander = None
    if expand_feautures is not None:
        expander = make_column_transformer(
            *[(ExpandFeatures(expand='categorical_interaction'),
               [name, expander_column])
              for name in names],
            ('drop', expander_column))

    preprocessor_ = preprocessor
    if preprocessor_ is None:
        preprocessor_ = StandardScaler()

    estimator_ = estimator
    if estimator_ is None:
        estimator_ = RidgeCV(alphas=np.logspace(-3, 5, 100))

 
    def _get_projector_vectorizer(projection, vectorization):
        return [(make_pipeline(*
                               [projection(**projection_params_),
                                vectorization(**vectorization_params_)]),
                 name) for name in names]
    
    # setup pipelines (projection + vectorization step)
    steps = tuple()
    if pipeline == 'riemann':
        steps = (ProjCommonSpace, Riemann)
    elif pipeline == 'lw_riemann':
        steps = (ProjLWSpace, Riemann)
    elif pipeline == 'diag':
        steps = (ProjIdentitySpace, Diag)
    elif pipeline == 'logdiag':
        steps = (ProjIdentitySpace, LogDiag)
    elif pipeline == 'random':
        steps = (ProjRandomSpace, LogDiag)
    elif pipeline == 'naive':
        steps = (ProjIdentitySpace, NaiveVec)
    elif pipeline == 'spoc':
        steps = (ProjSPoCSpace, LogDiag)
    elif pipeline == 'riemann_wasserstein':
        steps = (ProjIdentitySpace, RiemannSnp)

    pipeline_steps = [
        make_column_transformer(
            *_get_projector_vectorizer(*steps),
            remainder='passthrough'),
        preprocessor,
        estimator]

    if expander is not None:
        pipeline_steps.insert(1, expander)
    filter_bank = make_pipeline(*pipeline_steps)

    return filter_bank
