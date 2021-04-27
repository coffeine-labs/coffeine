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
        Xout = self.ts.transform(X)
        return pd.DataFrame(Xout)  # (sub, c*(c+1)/2)


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
        return pd.DataFrame(Xout)  # (sub,p)


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
        return pd.DataFrame(Xout)  # (sub,p)


class ExpandFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, expander_column):
        self.expander_column = expander_column
        self.estimator = estimator

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        self.estimator.fit(X.drop(self.expander_column, axis=1), y)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        indicator = X[self.expander_column].values[:, None]
        Xt = self.estimator.transform(X.drop(self.expander_column, axis=1))
        Xt = np.concatenate((Xt, indicator * Xt, indicator), axis=1)
        # (n, n_features + 1)
        return Xt


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
        return pd.DataFrame(Xout)  # (sub, p*(p+1)/2)


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
        return pd.DataFrame(Xout)  # (sub, c*(c+1)/2)


class Snp(TransformerMixin):
    def __init__(self, rank):
        """Init."""
        self.rank = rank

    def fit(self, X, y=None, ref=None):
        if ref is None:
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
