from typing import Union
import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def _check_data(X):
    # make proper 3d array of covariances
    out = None
    if X.ndim == 3:
        out = X
    elif X.values.dtype == 'object':
        # first remove unnecessary dimensions,
        # then stack to 3d data
        values = X.values
        if values.shape[1] == 1:
            values = values[:, 0]
        out = np.stack(values)
        if out.ndim == 2:  # deal with single sample
            assert out.shape[0] == out.shape[1]
            out = out[np.newaxis, :, :]
    return out

class NaiveVec(BaseEstimator, TransformerMixin):
    """Vectorize SPD matrix by flattening the upper triangle.

    Upper "naive" vectorization as described in [1]_.

    Parameters
    ----------
    metric : str, default='riemann'
        The Riemannian metric to use. See PyRiemann documentation for details
        and valid choices.
    return_data_frame : bool, default=True
        Returning the result in a pandas data frame or not.

    References
    ----------
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
           Predictive regression modeling with MEG/EEG: from source power
           to signals and cognitive states.
           *NeuroImage*, page 116893,2020. ISSN 1053-8119.
           https://doi.org/10.1016/j.neuroimage.2020.116893
    """
    def __init__(self, method, return_data_frame=True):
        self.method = method
        self.return_data_frame = return_data_frame
        return None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Extract vectorized upper triangle.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_data(X)
        n_sub, p, _ = X.shape
        q = int(p * (p+1) / 2)
        X_out = np.empty((n_sub, q))
        for sub in range(n_sub):
            if self.method == 'upper':
                X_out[sub] = X[sub][np.triu_indices(p)]
        if self.return_data_frame:
            X_out = pd.DataFrame(X_out)
        return X_out  # (sub, p*(p+1)/2)


class Diag(BaseEstimator, TransformerMixin):
    """Vectorize SPD matrix by extracting diagonal.

    This is equivalent of the M/EEG power spectrum in a given frequency bin.

    Parameters
    ----------
    metric : str, default='riemann'
        The Riemannian metric to use. See PyRiemann documentation for details
        and valid choices.
    return_data_frame : bool, default=True
        Returning the result in a pandas data frame or not.
    """
    def __init__(self, return_data_frame=True):
        self.return_data_frame = return_data_frame
        return None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Provide expected API for scikit-learn pipeline.

        .. note::
            The diagonal step does not fit any parameters.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Extract diagonal from X.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_data(X)
        n_sub, p, _ = X.shape
        X_out = np.empty((n_sub, p))
        for sub in range(n_sub):
            X_out[sub] = np.diag(X[sub])
        if self.return_data_frame:
            X_out = pd.DataFrame(X_out)
        return X_out  # (sub,p)


class LogDiag(BaseEstimator, TransformerMixin):
    """Vectorize SPD matrix by extracting diagonal and computing the log.

    log diagonal vectorization as described in [1]_.

    Parameters
    ----------
    metric : str, default='riemann'
        The Riemannian metric to use. See PyRiemann documentation for details
        and valid choices.
    return_data_frame : bool, default=True
        Returning the result in a pandas data frame or not.

    References
    ----------
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
           Predictive regression modeling with MEG/EEG: from source power
           to signals and cognitive states.
           *NeuroImage*, page 116893,2020. ISSN 1053-8119.
           https://doi.org/10.1016/j.neuroimage.2020.116893
    """
    def __init__(self, return_data_frame=True):
        self.return_data_frame = return_data_frame
        return None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Provide expected API for scikit-learn pipeline.

        .. note::
            The diagonal step does not fit any parameters.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Extract log diagonal from X.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_data(X)
        n_sub, p, _ = X.shape
        X_out = np.empty((n_sub, p))
        for sub in range(n_sub):
            X_out[sub] = np.diag(X[sub])
        np.log(X_out, out=X_out)
        if self.return_data_frame:
            X_out = pd.DataFrame(X_out)
        return X_out  # (sub,p)


class Riemann(BaseEstimator, TransformerMixin):
    """Map SPD matrix to Riemannian tangent space.

    Riemannian embedding step as described in [1]_.
    Implements affine invariant metric, which makes assumption of
    full-rank inputs.
    The transform implies a log non-linearity.

    Parameters
    ----------
    metric : str, default='riemann'
        The Riemannian metric to use. See PyRiemann documentation for details
        and valid choices.
    return_data_frame : bool, default=True
        Returning the result in a pandas data frame or not.

    References
    ----------
    .. [1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
           Predictive regression modeling with MEG/EEG: from source power
           to signals and cognitive states.
           *NeuroImage*, page 116893,2020. ISSN 1053-8119.
           https://doi.org/10.1016/j.neuroimage.2020.116893
    """
    def __init__(self, metric: str = 'riemann',
                 return_data_frame: bool = True):
        self.metric = metric
        self.return_data_frame = return_data_frame

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        X = _check_data(X)
        self.ts = TangentSpace(metric=self.metric).fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Project X to Riemannian tangent space defined by the training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_data(X)
        X_out = self.ts.transform(X)
        if self.return_data_frame:
            X_out = pd.DataFrame(X_out)
        return X_out  # (sub, c*(c+1)/2)


class RiemannSnp(BaseEstimator, TransformerMixin):
    """Map SPD matrix to Riemannian Wasserstein tangent space.

    Riemannian Wasserstein embedding step as described in [1]_.
    Implements Wasserstein metric that is not making a strong
    assumption of full-rank inputs.
    The transform implies a square-root non-linearity.

    Parameters
    ----------
    metric : str, default='riemann'
        The Riemannian metric to use. See PyRiemann documentation for details
        and valid choices.
    return_data_frame : bool, default=True
        Returning the result in a pandas data frame or not.

    References
    ----------
    .. [1] Sabbagh, D., Ablin, P., Varoquaux, G., Gramfort, A. and Engemann,
           D.A., 2019. Manifold-regression to predict from MEG/EEG brain
           signals without source modeling. Advances in Neural Information
           Processing Systems, 32.
    """
    def __init__(self, rank='full', return_data_frame=True):
        self.rank = rank
        self.return_data_frame = return_data_frame

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        X = _check_data(X)
        self.rank = len(X[0]) if self.rank == 'full' else self.rank
        self.ts = Snp(rank=self.rank).fit(X)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Project X to Riemannian SNP tangent space defined by training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_data(X)
        n_sub, p, _ = X.shape
        q = p * self.rank
        X_out = np.empty((n_sub, q))
        X_out = self.ts.transform(X)
        if self.return_data_frame:
            X_out = pd.DataFrame(X_out)
        return X_out  # (sub, c*(c+1)/2)


class Snp(TransformerMixin):
    """Map SPD matrix to Riemannian Wasserstein tangent space.

    Riemannian Wasserstein embedding step as described in [1]_.
    Implements Wasserstein metric that is not making the assumption
    of full-rank inputs.
    The transform implies a square-root non-linearity.

    Parameters
    ----------
    rank : int
        The rank to be used for sub-space projection.

    References
    ----------
    .. [1] Sabbagh, D., Ablin, P., Varoquaux, G., Gramfort, A. and Engemann,
           D.A., 2019. Manifold-regression to predict from MEG/EEG brain
           signals without source modeling. Advances in Neural Information
           Processing Systems, 32.
    """
    def __init__(self, rank: int):
        """Init."""
        self.rank = rank

    def fit(self,
            X: np.ndarray,
            y: Union[list[int, float], np.ndarray, None] = None,
            ref: Union[np.ndarray, None] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {np.ndarray} of shape (n_samples, n_channles, n_channels)
            Training vector, where `n_samples` is the number of samples.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        ref : np.ndarray of shape(n_channels, n_channels) or None
            A reference covaraiance. If None (default): arithmetic mean.
        """
        if ref is None:
            ref = np.mean(X, axis=0)
        Y = _to_quotient(ref, self.rank)
        self.reference_ = ref
        self.Y_ref_ = Y
        return self

    def transform(self, X: np.ndarray):
        """Project X to Riemannian SNP tangent space defined by training data.

        Parameters
        ----------
        X : {np.ndarray} of shape (n_samples, n_channles, n_channels)
            Training vector, where `n_samples` is the number of samples.
        """
        n_mat, n, _ = X.shape
        output = np.zeros((n_mat, n * self.rank))
        for j, C in enumerate(X):
            Y = _to_quotient(C, self.rank)
            output[j] = _logarithm(Y, self.Y_ref_).ravel()
        return output


def _to_quotient(C, rank):
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    Y = U * np.sqrt(d)
    return Y


def _logarithm(Y, Y_ref):
    prod = np.dot(Y_ref.T, Y)
    U, D, V = np.linalg.svd(prod, full_matrices=False)
    Q = np.dot(U, V).T
    return np.dot(Y, Q) - Y_ref


class ExpandFeatures(BaseEstimator, TransformerMixin):
    """Add binary interaction terms after projection step.

    Simple ad-hoc interaction features in projected space
    by multiplying a scaler (continuous or categorical) sample-level feature
    with the representation obtained from projection and vectorization, e.g.,
    drug dosage or biomarker value at baseline.

    Parameters
    ----------
    estimator : sklearn pipeline
        A coffeine filter-bank transformer, regressor or classifier.
    expander_column : str
        The column in the coffeine data frame (passed through as reminder)
        that should be used for computing the interaction features by
        multiplication.
    """
    def __init__(self, estimator: Pipeline, expander_column: str):
        self.expander_column = expander_column
        self.estimator = estimator

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        self.estimator.fit(X.drop(self.expander_column, axis=1), y)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Apply transform and add expanded features.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        indicator = X[self.expander_column].values[:, None]
        Xt = self.estimator.transform(X.drop(self.expander_column, axis=1))
        Xt = np.concatenate((Xt, indicator * Xt, indicator), axis=1)
        # (n, n_features + 1)
        return Xt
