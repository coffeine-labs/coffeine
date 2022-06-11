import copy as cp

import numpy as np
import pandas as pd
from mne import EvokedArray, Info
from scipy.linalg import eigh, pinv
from sklearn.base import BaseEstimator, TransformerMixin


def shrink(cov, alpha):
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.trace(cov) * np.eye(n) / n
    return shrink_cov


def fstd(y):
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
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xout = np.array(list(np.squeeze(X))).astype(float)
        return pd.DataFrame({'cov': list(Xout)})


class ProjCommonSpace(BaseEstimator, TransformerMixin):
    def __init__(self, scale='auto', n_compo='full', reg=1e-7):
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y=None):
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

    def transform(self, X):
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


class ProjLWSpace(BaseEstimator, TransformerMixin):
    def __init__(self, shrink):
        self.shrink = shrink

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _check_X_df(X)
        n_sub, p, _ = X.shape
        Xout = np.empty((n_sub, p, p))
        for sub in range(n_sub):
            Xout[sub] = shrink(X[sub], self.shrink)
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)


class ProjRandomSpace(BaseEstimator, TransformerMixin):
    def __init__(self, n_compo='full'):
        self.n_compo = n_compo

    def fit(self, X, y=None):
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        n_sub, n_chan, _ = X.shape
        U = np.linalg.svd(np.random.rand(n_chan, n_chan))[0][:self.n_compo]
        self.filter_ = U  # (compo, chan) row vec
        return self

    def transform(self, X):
        X = _check_X_df(X)
        n_sub = len(X)
        Xout = np.empty((n_sub, self.n_compo, self.n_compo))
        filter_ = self.filter_  # (compo, chan)
        for sub in range(n_sub):
            Xout[sub] = filter_ @ X[sub] @ filter_.T
        return pd.DataFrame({'cov': list(Xout)})  # (sub , compo, compo)


class ProjSPoCSpace(BaseEstimator, TransformerMixin):
    def __init__(self, shrink=0, scale=1, n_compo='full', reg=1e-7):
        self.shrink = shrink
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y=None):
        X = _check_X_df(X)
        self.n_compo = len(X[0]) if self.n_compo == 'full' else self.n_compo
        target = fstd(y)
        self.scale_ = _get_scale(X, self.scale)
        C = X.mean(axis=0)
        Cz = np.mean(X * target[:, None, None], axis=0)
        C = shrink(C, self.shrink)
        eigvals, eigvecs = eigh(Cz, C)
        ix = np.argsort(np.abs(eigvals))[::-1]
        evecs = eigvecs[:, ix]
        evecs = evecs[:, :self.n_compo].T
        evecs /= np.linalg.norm(evecs, axis=1)[:, None]
        self.filter_ = evecs  # (compo, chan) row vec
        self.pattern_ = pinv(evecs).T  # (compo, chan)
        return self

    def transform(self, X):
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
                      show=True, show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None,
                      axes=None):

        if components is None:
            components = np.arange(self.n_compo)
        pattern = self.pattern_

        # set sampling frequency to have 1 component per time point
        info_ = dict(info)
        info_['sfreq'] = 1.
        info = Info(**info_)
        norm_pattern = pattern / np.linalg.norm(pattern, axis=1)[:, None]
        pattern_array = EvokedArray(norm_pattern.T, info, tmin=0)
        return pattern_array.plot_topomap(
            times=components, ch_type=ch_type,
            vmin=vmin, vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, axes=axes)

    def plot_filters(self, info, components=None,
                     ch_type=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, axes=None):

        if components is None:
            components = np.arange(self.n_compo)
        filter_ = self.filter_

        # set sampling frequency to have 1 component per time point
        info_ = dict(info)
        info_['sfreq'] = 1.
        info = Info(**info_)
        filter_array = EvokedArray(filter_, info, tmin=0)
        return filter_array.plot_topomap(
            times=components, ch_type=ch_type, vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, title=title, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average,
            axes=axes)
