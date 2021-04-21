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
        fb_cols=('low', 'delta', 'theta', 'alpha', 'beta'),
        scale='auto',
        n_compo='full',
        reg=1.e-05,
        metric='riemann',
        shrink=1,
        rank='full',  # for RiemannWass
        method='upper',  # for NaiveVec: 'upper' 'upperlog' 'logdiag+upper'
        expand=False,
        # if expand=True, learns the interaction effect w/ last binary column
        ridge_alphas=np.logspace(-3, 5, 100)):

    # Define intermediate Transformers
    vec_riemann = make_column_transformer(
        *[(make_pipeline(ProjCommonSpace(scale=scale, n_compo=n_compo, reg=reg),
                         Riemann(metric=metric)), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_lw_riemann = make_column_transformer(
        *[(make_pipeline(ProjLWSpace(shrink=shrink),
                         Riemann(metric=metric)), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_diag = make_column_transformer(
        *[(make_pipeline(ProjIdentitySpace(),
                         Diag()), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_logdiag = make_column_transformer(
        *[(make_pipeline(ProjIdentitySpace(),
                         LogDiag()),col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_random = make_column_transformer(
        *[(make_pipeline(ProjRandomSpace(n_compo=n_compo),
                        LogDiag()), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_naive = make_column_transformer(
        *[(make_pipeline(ProjIdentitySpace(),
                         NaiveVec(method=method)), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_spoc = make_column_transformer(
        *[(make_pipeline(ProjSPoCSpace(n_compo=n_compo, scale=scale, reg=reg,
                                       shrink=shrink),
                         LogDiag()), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    vec_riemann_wass = make_column_transformer(
        *[(make_pipeline(ProjIdentitySpace(),
                         RiemannSnp(rank=rank)), col)
          for col in fb_cols],
        remainder='passthrough'
    )
    expander = make_column_transformer(
        *[(ExpandFeatures(expand=expand), [col, -1])
          for col in range(len(fb_cols))],
        ('drop', -1)
    )

    # Define pipelines
    pipelines = {
        'riemann': make_pipeline(
            vec_riemann,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'lw_riemann': make_pipeline(
            vec_lw_riemann,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'diag': make_pipeline(
            vec_diag,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'logdiag': make_pipeline(
            vec_logdiag,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'random': make_pipeline(
            vec_random,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'naive': make_pipeline(
            vec_naive,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'spoc': make_pipeline(
            vec_spoc,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'riemann_wass': make_pipeline(
            vec_riemann_wass,
            expander,
            StandardScaler(),
            RidgeCV(alphas=ridge_alphas)),
        'dummy': make_pipeline(
            vec_logdiag,
            expander,
            StandardScaler(),
            DummyRegressor())
    }
    return pipelines
