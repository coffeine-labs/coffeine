from typing import Union
import numpy as np
import pandas as pd
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

from coffeine.transfer_learning import (
    ReCenter,
    ReScale
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression


class GaussianKernel(BaseEstimator, TransformerMixin):
    """Gaussian (squared exponential) Kernel.

    Efficient computation of squared exponential kernel for
    one column of covariances in a coffeine DataFrame.

    Parameters
    ----------
    sigma : float
        The sigma or length-scale parameter of the Gaussian kernel.
    """
    def __init__(self, sigma: float = 1.):
        self.sigma = sigma

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Prepare fitting kernel on training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_covariances` = 1 (inside a column of a coffeine data frame).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = X.astype(np.float64)
        self.N = np.sum(self.X ** 2, axis=1)
        return self

    def transform(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[list[int, float], np.ndarray, None] = None):
        """Compute Kernel.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_covariances` = 1 (inside a column of a coffeine data frame).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        C = 1.
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_d = X.astype(np.float64)

        # compute L2 norm across feature vectors in every subject
        N = np.sum(X_d ** 2, axis=1)
        C1 = self.N[None, :] + N[:, None]
        C2 = (X_d.reshape(X_d.shape[0], -1) @
              self.X.reshape(self.X.shape[0], -1).T)
        C = np.exp(-(C1 - 2 * C2) / (self.sigma ** 2))

        return C

    def get_params(self, deep: bool = True):
        """Get parameters."""
        return {"sigma": self.sigma}

    def set_params(self, **parameters):
        """Get parameters."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class KernelSum(BaseEstimator, TransformerMixin):
    """Sum multiple Kernels with fixed equal weights.

    Expects input from concatenated output of ColumnTransformer in
    a filterbank pipeline.
    """
    def __init__(self):
        pass

    def fit(self,
            X: np.ndarray,
            y: Union[list[int, float], np.ndarray, None] = None):
        """Implement API neede for scikit-learn pipeline.

        Parameters
        ----------
        X : {np.array} of shape (n_samples, n_covariances * n_samples_train)
            Training vector, where `n_samples` is the number of samples and
            `n_covariances` = 1 (inside a column of a coffeine data frame).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        self.n_train_ = len(X)
        return self

    def transform(self,
                  X: np.ndarray,
                  y: Union[list[int, float], np.ndarray, None] = None):
        """Sum various kernels returned by column transformer.

        Parameters
        ----------
        X : {np.array} of shape (n_samples, n_covariances * n_samples_train)
            Training vector, where `n_samples` is the number of samples and
            `n_covariances` = 1 (inside a column of a coffeine data frame).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        X_out = X
        if X.shape not in ((len(X), self.n_train_), (len(X), len(X))):
            X_out = X.reshape(len(X), -1, self.n_train_).sum(axis=1)
        return X_out


def make_filter_bank_transformer(
        names: list[str],
        method: str = 'riemann',
        alignment: Union[list[str], None] = None,
        domains: Union[list[str], None] = None,
        projection_params: Union[dict, None] = None,
        vectorization_params: Union[dict, None] = None,
        kernel: Union[str, Pipeline, None] = None,
        combine_kernels: Union[str, Pipeline, None] = None,
        categorical_interaction: Union[bool, None] = None):
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
    alignment : list of str | None
        Alignment steps to include in the pipeline. Can be ``'re-center'``,
        ``'re-scale'``.
    projection_params : dict | None
        The parameters for the projection step.
    vectorization_params : dict | None
        The parameters for the vectorization step.
    kernel : None | 'gaussian' | sklearn.Pipeline
        The Kernel option for kernel regression. If 'gaussian', a Gaussian
        Kernel will be added per column and the results will be summed over
        frequencies. If sklearn.pipeline.Pipeline is passed, it should return
        a meaningful kernel.
    combine_kernels : None | 'sum' | sklearn.pipeline.Pipeline
        If kernel is used and multiple columns are defined, this option
        determines how a combined kernel is constructed. 'sum' adds the
        kernels with equal weights. A custom pipeline pipeline can be passed to
        implement alternative rules.
    categorical_interaction : str
        The column in the input data frame containing a binary descriptor
        used to fit 2-way interaction effects.

    References
    ----------
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
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

    def _get_projector_vectorizer(projection, vectorization,
                                  alignment_steps=None,
                                  kernel=None):
        out = list()
        for name in names:
            if alignment_steps is None:
                steps = [
                    projection(**projection_params_),
                    vectorization(**vectorization_params_)
                ]
            else:
                steps = [
                    projection(**projection_params_)
                ] + alignment_steps + [
                    vectorization(**vectorization_params_)
                ]
            if kernel is not None:
                kernel_name, kernel_estimator = kernel
                steps.append(kernel_estimator())
            pipeline = make_pipeline(*steps)
            out.append((pipeline, name))
        return out

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

    # add alignment options
    alignment_steps = []
    if alignment is None:
        alignment_steps = None
    else:
        if 're-center' in alignment:
            alignment_steps.append(ReCenter(domains=domains))
        if 're-scale' in alignment:
            alignment_steps.append(ReScale(domains=domains))

    # add Kernel options
    if (isinstance(kernel, Pipeline) and not
            isinstance(kernel, (BaseEstimator, TransformerMixin))):
        raise ValueError(
            'Custom kernel must be an estimator and a transformer).'
        )
    elif kernel == 'gaussian':
        kernel = (
            'gaussiankernel', GaussianKernel
        )
        combine_kernels = 'sum'

    filter_bank_transformer = make_column_transformer(
        *_get_projector_vectorizer(*steps, alignment_steps=alignment_steps,
                                   kernel=kernel),
        remainder='passthrough'
    )

    if combine_kernels is not None:
        filter_bank_transformer = make_pipeline(
            filter_bank_transformer,
            KernelSum() if combine_kernels == 'sum' else combine_kernels
        )
    if categorical_interaction is not None:
        filter_bank_transformer = ExpandFeatures(
            filter_bank_transformer, expander_column=categorical_interaction)

    return filter_bank_transformer


def make_filter_bank_regressor(
        names: list[str],
        method: str = 'riemann',
        projection_params: Union[dict, None] = None,
        vectorization_params: Union[dict, None] = None,
        categorical_interaction: Union[bool, None] = None,
        scaling: Union[BaseEstimator, None] = None,
        estimator: Union[BaseEstimator, None] = None):
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
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
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


def make_filter_bank_classifier(
        names: list[str],
        method: str = 'riemann',
        projection_params: Union[dict, None] = None,
        vectorization_params: Union[dict, None] = None,
        categorical_interaction: Union[bool, None] = None,
        scaling: Union[BaseEstimator, None] = None,
        estimator: Union[BaseEstimator, None] = None):
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
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
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

    filter_bank_classifier = make_pipeline(
        filter_bank_transformer,
        scaling_,
        estimator_
    )

    return filter_bank_classifier
