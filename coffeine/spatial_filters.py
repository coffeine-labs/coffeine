from typing import Union
import numpy as np
import pandas as pd
from mne import EvokedArray
from scipy.linalg import eigh, pinv
from sklearn.base import BaseEstimator, TransformerMixin


def _shrink(cov, alpha):
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.trace(cov) * np.eye(n) / n
    return shrink_cov


def _fstd(y):
    y = y.astype(np.float32)
    y -= y.mean(axis=0)
    y /= y.std(axis=0)
    return y


def _get_scale(X, scale):
    if scale == 'auto':
        scale = 1 / np.mean([np.trace(x) for x in X])
    return scale


def _check_X_df(X):
    if hasattr(X, 'values'):
        X = np.array(list(np.squeeze(X))).astype(float)
        if X.ndim == 2:  # deal with single sample
            assert X.shape[0] == X.shape[1]
            X = X[np.newaxis, :, :]
    return X

class ProjIdentitySpace(BaseEstimator, TransformerMixin):
    """Apply identy projection to SPD matrix.

    Helper to skip projection step.
    """
    def __init__(self):
        return None

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Provide expected API for scikit-learn pipeline.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        return self

    def transform(self, X):
        """Apply identity projection to X.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_X_df(X)
        Xout = np.array(list(np.squeeze(X))).astype(float)
        return pd.DataFrame({'cov': list(Xout)})


class ProjRandomSpace(BaseEstimator, TransformerMixin):
    """Apply random projection to SPD matrix.

    Asses chance-level at projection step via random projections.

    Parameters
    ----------
    n_compo : int
        The size of the subspace to project onto.
    random_state : int | np.random.RandomState
        The random state.
    """
    def __init__(self, n_compo: str = 'full',
                 random_state: Union[int, np.random.RandomState] = 42):
        self.n_compo = n_compo
        if isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state

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
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        _, n_chan, _ = X.shape
        U = np.linalg.svd(
            self.random_state.rand(n_chan, n_chan))[0][:self.n_compo]
        self.filter_ = U  # (compo, chan) row vec
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Apply random projection defined at training time.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_X_df(X)
        n_sub = len(X)
        Xout = np.empty((n_sub, self.n_compo, self.n_compo))
        filter_ = self.filter_  # (compo, chan)
        for sub in range(n_sub):
            Xout[sub] = filter_ @ X[sub] @ filter_.T
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)


class ProjCommonSpace(BaseEstimator, TransformerMixin):
    """Project SPD matrix to common subspace (PCA).

    Needed to define Riemannian metrics with rank deficient inputs as
    described in [1]_.

    Parameters
    ----------
    n_compo : int
        The size of the subspace to project onto.
    random_state : float | np.random.RandomState
        The random state.
    scale : float | 'auto'
        Optional normalization that can be applied to the covariance.
        If float, the value is directly used as a scaling factor. If auto,
        scaling is obtained by 1 devided by the average of the trace across
        covariances.
    reg : float (defaults to 1e-15)
        A regularization factor applied in subspace by reg * identity matrix.
        The number is chose to be small and numerically stabilizing, assuming
        EEG input scaled in volts. This is sensitive to the scale of the input
        and may be different for MEG or other data types. Please check.

    References
    ----------
    .. [1] Sabbagh, D., Ablin, P., Varoquaux, G., Gramfort, A. and Engemann,
           D.A., 2019. Manifold-regression to predict from MEG/EEG brain
           signals without source modeling. Advances in Neural Information
           Processing Systems, 32.
    """
    def __init__(self, scale: float = 1., n_compo: Union[int, str] = 'full',
                 reg: float = 1e-15):
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Compute filters for subspace projection given the training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        C = X.mean(axis=0)
        eigvals, eigvecs = eigh(C)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :self.n_compo].T
        self.filters_.append(evecs)  # (fb, compo, chan) row vec
        self.patterns_.append(pinv(evecs).T)  # (fb, compo, chan)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Project X to subspace using the filters defined on training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_X_df(X)
        n_sub, _, _ = X.shape
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        Xout = np.empty((n_sub, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        filters = self.filters_[0]  # (compo, chan)
        for sub in range(n_sub):
            Xout[sub] = filters @ Xs[sub] @ filters.T
            Xout[sub] += self.reg * np.eye(self.n_compo)
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)


class ProjSPoCSpace(BaseEstimator, TransformerMixin):
    """Project SPD matrix subspace given by SPoC.

    Computes Source Power Co-Modulation SPoC presented in [1]_.

    .. note::
        This implementation is absed on MNE-Python.
        Contrary to the PyRiemann implementation, this implementation use
        the arithmetic mean across the covariances as a reference.

    Parameters
    ----------
    n_compo : int
        The size of the subspace to project onto.
    random_state : float | np.random.RandomState
        The random state.
    shrink : float
        The shrinkage factor, like alpha in scikit-learn `shrunk_covariance`.
    scale : float | 'auto'
        Optional normalization that can be applied to the covariance.
        If float, the value is directly used as a scaling factor. If auto,
        scaling is obtained by 1 devided by the average of the trace across
        covariances.
    reg : float (defaults to 1e-15)
        A regularization factor applied in subspace by reg * identity matrix.
        The number is chose to be small and numerically stabilizing, assuming
        EEG input scaled in volts. This is sensitive to the scale of the input
        and may be different for MEG or other data types. Please check.

    References
    ----------
    .. [1] Dähne, S., Meinecke, F.C., Haufe, S., Höhne, J., Tangermann, M.,
           Müller, K.R. and Nikulin, V.V., 2014. SPoC: a novel framework for
           relating the amplitude of neuronal oscillations to behaviorally
           relevant parameters. NeuroImage, 86, pp.111-122.
    """
    def __init__(self, shrink: float = 0., scale: float = 1.,
                 n_compo: Union[int, str] = 'full', reg: float = 1e-15):
        self.shrink = shrink
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Compute filters for subspace projection given the training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        """
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        target = _fstd(y)
        self.scale_ = _get_scale(X, self.scale)
        C = X.mean(axis=0)
        Cz = np.mean(X * target[:, None, None], axis=0)
        C = _shrink(C, self.shrink)
        eigvals, eigvecs = eigh(Cz, C)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :self.n_compo].T
        evecs /= np.linalg.norm(evecs, axis=1)[:, None]
        self.filter_ = evecs  # (compo, chan) row vec
        self.pattern_ = pinv(evecs).T  # (compo, chan)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        """Project X to subspace using the filters defined on training data.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_X_df(X)
        n_sub = len(X)
        Xout = np.empty((n_sub, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        filter_ = self.filter_  # (compo, chan)
        for sub in range(n_sub):
            Xout[sub] = filter_ @ Xs[sub] @ filter_.T
            Xout[sub] += self.reg * np.eye(self.n_compo)
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)

    def plot_patterns(self, info, components=None,
                      ch_type=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scalings=None, units='a.u.', res=64,
                      size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                      show=True, show_names=False, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='cubic', average=None,
                      axes=None):
        """"Plot topographic patterns of components (inverse of filters).

        For detailed documentaiton, check out the MNE documentation

        """
        if components is None:
            components = np.arange(self.n_compo)
        pattern = self.pattern_

        # set sampling frequency to have 1 component per time point
        with info._unlock():
            info['sfreq'] = 1.
        norm_pattern = pattern / np.linalg.norm(pattern, axis=1)[:, None]
        pattern_array = EvokedArray(norm_pattern.T, info, tmin=0)
        return pattern_array.plot_topomap(
            times=components, ch_type=ch_type,
            vlim=(vmin, vmax), cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, axes=axes)

    def plot_filters(self, info, components=None,
                     ch_type=None,
                     vlim=(None, None), cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='cubic', average=None, axes=None):
        """"Plot topographic patterns of filters.

        For detailed documentaiton, check out the MNE documentation

        """
        if components is None:
            components = np.arange(self.n_compo)
        filter_ = self.filter_

        # set sampling frequency to have 1 component per time point
        with info._unlock():
            info['sfreq'] = 1.
        filter_array = EvokedArray(filter_, info, tmin=0)
        return filter_array.plot_topomap(
            times=components, ch_type=ch_type, vlim=vlim,
            cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average,
            axes=axes)


class ProjLWSpace(BaseEstimator, TransformerMixin):
    """Apply regularization on covariance matrices.

    A James-Stein type shrinkage is applied by weighting down
    the off-diagonal (cross) terms proportional to a shrinkage factor.

    Parameters
    ----------
    shrink : float
        The shrinkage factor, like alpha in scikit-learn `shrunk_covariance`.
    """
    def __init__(self, shrink: float):
        self.shrink = shrink

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[list[int, float], np.ndarray, None] = None):
        """Provide expected API for scikit-learn pipeline.

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
        """Apply shrinkage implied by pre-specified shinkage faxtor.

        Parameters
        ----------
        X : {pd.DataFrame} of shape (n_samples, n_covariances)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of covariances (inside the columns).
        """
        X = _check_X_df(X)
        n_sub, p, _ = X.shape
        Xout = np.empty((n_sub, p, p))
        for sub in range(n_sub):
            Xout[sub] = _shrink(X[sub], self.shrink)
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)
