import numpy as np
import pandas as pd
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin


class Riemann(BaseEstimator, TransformerMixin):
    def __init__(self, metric='wasserstein'):
        self.metric = metric

    def fit(self, X, y=None):
        X = np.array(list(np.squeeze(X)))
        self.ts = TangentSpace(metric=self.metric).fit(X)
        return self

    def transform(self, X):
        X = np.array(list(np.squeeze(X)))
        n_sub, p, _ = X.shape
        Xout = np.empty((n_sub, p*(p+1)//2))
        Xout = self.ts.transform(X)
        return pd.DataFrame({'cov': list(Xout.reshape(n_sub, -1))})
        # (sub, c*(c+1)/2)


class Diag(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(list(np.squeeze(X)))
        n_sub, p, _ = X.shape
        Xout = np.empty((n_sub, p))
        for sub in range(n_sub):
            Xout[sub] = np.diag(X[sub])
        return pd.DataFrame({'cov': list(Xout.reshape(n_sub, -1))})  # (sub,p)


class LogDiag(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(list(np.squeeze(X)))
        n_sub, p, _ = X.shape
        Xout = np.empty((n_sub, p))
        for sub in range(n_sub):
            Xout[sub] = np.log10(np.diag(X[sub]))
        return pd.DataFrame({'cov': list(Xout.reshape(n_sub, -1))})  # (sub,p)


class ExpandFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, expand=False):
        self.expand = expand

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = np.array(list(np.squeeze(X[:, :-1])))
        if self.expand is True:
            indicator = np.array(X[:, -1])[:, None]
            X = np.array(list(np.squeeze(X[:, :-1])))
            indicatorxX = indicator @ indicator.T @ X
            res = np.concatenate((X, indicator, indicatorxX), axis=1)
        return res


class NaiveVec(BaseEstimator, TransformerMixin):
    def __init__(self, method):
        self.method = method
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(list(np.squeeze(X)))
        n_sub, p, _ = X.shape
        q = int(p * (p+1) / 2)
        Xout = np.empty((n_sub, q))
        for sub in range(n_sub):
            if self.method == 'upper':
                Xout[sub] = X[sub][np.triu_indices(p)]
        return pd.DataFrame({'cov': list(Xout.reshape(n_sub, -1))})
        # (sub,p*(p+1)/2)


class RiemannSnp(BaseEstimator, TransformerMixin):
    def __init__(self, rank='full'):
        self.rank = rank

    def fit(self, X, y=None):
        X = np.array(list(np.squeeze(X)))
        self.rank = len(X[0]) if self.rank == 'full' else self.rank
        self.ts = Snp(rank=self.rank).fit(X)
        return self

    def transform(self, X):
        X = np.array(list(np.squeeze(X)))
        n_sub, p, _ = X.shape
        q = p * self.rank
        Xout = np.empty((n_sub, q))
        Xout = self.ts.transform(X)
        return pd.DataFrame({'cov': list(Xout.reshape(n_sub, -1))})
        # (sub, c*(c+1)/2)


class Snp(TransformerMixin):
    def __init__(self, rank):
        """Init."""
        self.rank = rank

    def fit(self, X, y=None, ref=None):
        if ref is None:
            #  ref = mean_covs(X, rank=self.rank)
            ref = np.mean(X, axis=0)
        Y = to_quotient(ref, self.rank)
        self.reference_ = ref
        self.Y_ref_ = Y
        return self

    def transform(self, X, verbose=False):
        n_mat, n, _ = X.shape
        output = np.zeros((n_mat, n * self.rank))
        for j, C in enumerate(X):
            Y = to_quotient(C, self.rank)
            output[j] = logarithm_(Y, self.Y_ref_).ravel()
        return output


def to_quotient(C, rank):
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    Y = U * np.sqrt(d)
    return Y


def logarithm_(Y, Y_ref):
    prod = np.dot(Y_ref.T, Y)
    U, D, V = np.linalg.svd(prod, full_matrices=False)
    Q = np.dot(U, V).T
    return np.dot(Y, Q) - Y_ref
