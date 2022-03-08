import numpy as np
from coffeine.covariance_transformers import (
    Diag,
    LogDiag,
    ExpandFeatures,
    Riemann,
    RiemannSnp,
    NaiveVec)

from coffeine.spatial_filters import (
    ProjIdentitySpace,
    ProjCommonSpace,
    ProjLWSpace,
    ProjRandomSpace,
    ProjSPoCSpace)

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression


def make_filter_bank_transformer(names, method='riemann',
                                 projection_params=None,
                                 vectorization_params=None,
                                 categorical_interaction=None):
    """Generate pipeline for filterbank models.

    Prepare filter bank models as used in [1]_. These models take as input
    sensor-space covariance matrices computed from M/EEG signals in different
    frequency bands. Then transformations are applied to improve the
    applicability of linear regression techniques by reducing the impact of
    field spread.

    In terms of implementation, this involves 1) projection
    (e.g. spatial filters) and 2) vectorization (e.g. taking the log on the
    diagonal).

    .. note::
        The resulting model expects as inputs data frames in which different
        covarances (e.g. for different frequencies) are stored inside columns
        indexed by ``names``.

        Other columns will be passed through by the underlying column
        transformers.

        The pipeline also supports fitting categorical interaction effects
        after projection and vectorization steps are performed.

    .. note::
        All essential methods from [1]_ are implemented here. In practice,
        we recommend comparing `riemann', `spoc' and `diag' as a baseline.

    Parameters
    ----------
    names : list of str
        The column names of the data frame corresponding to different
        covariances.
    method : str
        The method used for extracting features from covariances. Defaults
        to ``'riemann'``. Can be ``'riemann'``, ``'lw_riemann'``, ``'diag'``,
        ``'log_diag'``, ``'random'``, ``'naive'``, ``'spoc'``,
        ``'riemann_wasserstein'``.
    projection_params : dict | None
        The parameters for the projection step.
    vectorization_params : dict | None
        The parameters for the vectorization step.
    categorical_interaction : str
        The column in the input data frame containing a binary descriptor
        used to fit 2-way interaction effects.

    References
    ----------
    [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
        Predictive regression modeling with MEG/EEG: from source power
        to signals and cognitive states.
        *NeuroImage*, page 116893,2020. ISSN 1053-8119.
        https://doi.org/10.1016/j.neuroimage.2020.116893

    """
    # put defaults here for projection and vectorization step
    projection_defaults = {
        'riemann': dict(scale='auto', n_compo='full', reg=1.e-05),
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

    assert set(projection_defaults) == set(vectorization_defaults)

    if method not in projection_defaults:
        raise ValueError(
            f"The `method` ('{method}') you specified is unknown.")

    # update defaults
    projection_params_ = projection_defaults[method]
    if projection_params is not None:
        projection_params_.update(**projection_params)

    vectorization_params_ = vectorization_defaults[method]
    if vectorization_params is not None:
        vectorization_params_.update(**vectorization_params)

    def _get_projector_vectorizer(projection, vectorization):
        return [(make_pipeline(*
                               [projection(**projection_params_),
                                vectorization(**vectorization_params_)]),
                 name) for name in names]

    # setup pipelines (projection + vectorization step)
    steps = tuple()
    if method == 'riemann':
        steps = (ProjCommonSpace, Riemann)
    elif method == 'lw_riemann':
        steps = (ProjLWSpace, Riemann)
    elif method == 'diag':
        steps = (ProjIdentitySpace, Diag)
    elif method == 'log_diag':
        steps = (ProjIdentitySpace, LogDiag)
    elif method == 'random':
        steps = (ProjRandomSpace, LogDiag)
    elif method == 'naive':
        steps = (ProjIdentitySpace, NaiveVec)
    elif method == 'spoc':
        steps = (ProjSPoCSpace, LogDiag)
    elif method == 'riemann_wasserstein':
        steps = (ProjIdentitySpace, RiemannSnp)

    filter_bank_transformer = make_column_transformer(
        *_get_projector_vectorizer(*steps), remainder='passthrough')

    if categorical_interaction is not None:
        filter_bank_transformer = ExpandFeatures(
            filter_bank_transformer, expander_column=categorical_interaction)

    return filter_bank_transformer


def make_filter_bank_regressor(names, method='riemann',
                               projection_params=None,
                               vectorization_params=None,
                               categorical_interaction=None, scaling=None,
                               estimator=None):
    """Generate pipeline for regression with filter bank model.

    Prepare filter bank models as used in [1]_. These models take as input
    sensor-space covariance matrices computed from M/EEG signals in different
    frequency bands. Then transformations are applied to improve the
    applicability of linear regression techniques by reducing the impact of
    field spread.

    In terms of implementation, this involves 1) projection
    (e.g. spatial filters) and 2) vectorization (e.g. taking the log on the
    diagonal).

    .. note::
        The resulting model expects as inputs data frames in which different
        covarances (e.g. for different frequencies) are stored inside columns
        indexed by ``names``.

        Other columns will be passed through by the underlying column
        transformers.

        The pipeline also supports fitting categorical interaction effects
        after projection and vectorization steps are performed.

    .. note::
        All essential methods from [1]_ are implemented here. In practice,
        we recommend comparing `riemann', `spoc' and `diag' as a baseline.

    Parameters
    ----------
    names : list of str
        The column names of the data frame corresponding to different
        covariances.
    method : str
        The method used for extracting features from covariances. Defaults
        to ``'riemann'``. Can be ``'riemann'``, ``'lw_riemann'``, ``'diag'``,
        ``'log_diag'``, ``'random'``, ``'naive'``, ``'spoc'``,
        ``'riemann_wasserstein'``.
    projection_params : dict | None
        The parameters for the projection step.
    vectorization_params : dict | None
        The parameters for the vectorization step.
    categorical_interaction : str
        The column in the input data frame containing a binary descriptor
        used to fit 2-way interaction effects.
    scaling : scikit-learn Transformer object | None
        Method for re-rescaling the features. Defaults to None. If None,
        StandardScaler is used.
    estimator : scikit-learn Estimator object.
        The estimator object. Defaults to None. If None, RidgeCV
        is performed with default values.

    References
    ----------
    [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
        Predictive regression modeling with MEG/EEG: from source power
        to signals and cognitive states.
        *NeuroImage*, page 116893,2020. ISSN 1053-8119.
        https://doi.org/10.1016/j.neuroimage.2020.116893

    """
    filter_bank_transformer = make_filter_bank_transformer(
        names=names, method=method, projection_params=projection_params,
        vectorization_params=vectorization_params,
        categorical_interaction=categorical_interaction
    )

    scaling_ = scaling
    if scaling_ is None:
        scaling_ = StandardScaler()

    estimator_ = estimator
    if estimator_ is None:
        estimator_ = RidgeCV(alphas=np.logspace(-3, 5, 100))

    filter_bank_regressor = make_pipeline(
        filter_bank_transformer,
        scaling_,
        estimator_
    )

    return filter_bank_regressor


def make_filter_bank_classifier(names, method='riemann',
                                projection_params=None,
                                vectorization_params=None,
                                categorical_interaction=None, scaling=None,
                                estimator=None):
    """Generate pipeline for classification with filter bank model.

    Prepare filter bank models as used in [1]_. These models take as input
    sensor-space covariance matrices computed from M/EEG signals in different
    frequency bands. Then transformations are applied to improve the
    applicability of linear regression techniques by reducing the impact of
    field spread.

    In terms of implementation, this involves 1) projection
    (e.g. spatial filters) and 2) vectorization (e.g. taking the log on the
    diagonal).

    .. note::
        The resulting model expects as inputs data frames in which different
        covarances (e.g. for different frequencies) are stored inside columns
        indexed by ``names``.

        Other columns will be passed through by the underlying column
        transformers.

        The pipeline also supports fitting categorical interaction effects
        after projection and vectorization steps are performed.

    .. note::
        All essential methods from [1]_ are implemented here. In practice,
        we recommend comparing `riemann', `spoc' and `diag' as a baseline.

    Parameters
    ----------
    names : list of str
        The column names of the data frame corresponding to different
        covariances.
    method : str
        The method used for extracting features from covariances. Defaults
        to ``'riemann'``. Can be ``'riemann'``, ``'lw_riemann'``, ``'diag'``,
        ``'log_diag'``, ``'random'``, ``'naive'``, ``'spoc'``,
        ``'riemann_wasserstein'``.
    projection_params : dict | None
        The parameters for the projection step.
    vectorization_params : dict | None
        The parameters for the vectorization step.
    categorical_interaction : str
        The column in the input data frame containing a binary descriptor
        used to fit 2-way interaction effects.
    scaling : scikit-learn Transformer object | None
        Method for re-rescaling the features. Defaults to None. If None,
        StandardScaler is used.
    estimator : scikit-learn Estimator object.
        The estimator object. Defaults to None. If None, LogisticRegression
        is performed with default values.

    References
    ----------
    [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
        Predictive regression modeling with MEG/EEG: from source power
        to signals and cognitive states.
        *NeuroImage*, page 116893,2020. ISSN 1053-8119.
        https://doi.org/10.1016/j.neuroimage.2020.116893

    """
    filter_bank_transformer = make_filter_bank_transformer(
        names=names, method=method, projection_params=projection_params,
        vectorization_params=vectorization_params,
        categorical_interaction=categorical_interaction
    )

    scaling_ = scaling
    if scaling_ is None:
        scaling_ = StandardScaler()

    estimator_ = estimator
    if estimator_ is None:
        estimator_ = LogisticRegression(solver='liblinear')

    filter_bank_regressor = make_pipeline(
        filter_bank_transformer,
        scaling_,
        estimator_
    )

    return filter_bank_regressor
