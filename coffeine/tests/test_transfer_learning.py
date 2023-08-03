import pytest
import numpy as np
from pyriemann.datasets import make_classification_transfer
from pyriemann.transfer import decode_domains
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance

from coffeine.transfer_learning import ReCenter, ReScale


def test_recenter():
    n_matrices = 200
    domain_sep = 5
    X, y_enc = make_classification_transfer(n_matrices, domain_sep=domain_sep,
                                            random_state=42)
    _, y, domains = decode_domains(X, y_enc)
    train_index = [
        i for i in range(len(domains)) if domains[i] != 'target_domain'
    ]
    test_index = [
        i for i in range(len(domains)) if domains[i] == 'target_domain'
    ]
    X_train, y_train = X[train_index], y[train_index]
    X_test = X[test_index]
    rct = ReCenter(domains[train_index], metric='riemann')
    X_train_rct = rct.fit_transform(X_train, y_train)
    X_test_rct = rct.transform(X_test)
    # Test if mean is Identity
    M_train = mean_covariance(X_train_rct, metric='riemann')
    assert M_train == pytest.approx(np.eye(2))
    M_test = mean_covariance(X_test_rct, metric='riemann')
    assert M_test == pytest.approx(np.eye(2))


def test_rescale():
    n_matrices = 100
    stretch = 3
    X, y_enc = make_classification_transfer(n_matrices, stretch=stretch,
                                            random_state=42)
    _, y, domains = decode_domains(X, y_enc)
    train_index = [
        i for i in range(len(domains)) if domains[i] != 'target_domain'
    ]
    test_index = [
        i for i in range(len(domains)) if domains[i] == 'target_domain'
    ]
    X_train, y_train = X[train_index], y[train_index]
    X_test = X[test_index]
    str = ReScale(domains[train_index], metric='riemann')
    X_train_str = str.fit_transform(X_train, y_train)
    X_test_str = str.transform(X_test)
    # Test if dispersion = 1
    M_train = mean_covariance(X_train_str, metric='riemann')
    disp_train = np.sum(
        distance(X_train_str, M_train, metric='riemann')**2
    ) / X_train_str.shape[0]
    assert np.isclose(disp_train, 1.0)
    M_test = mean_covariance(X_test_str, metric='riemann')
    disp_test = np.sum(
        distance(X_test_str, M_test, metric='riemann')**2
    ) / X_test_str.shape[0]
    assert np.isclose(disp_test, 1.0)
