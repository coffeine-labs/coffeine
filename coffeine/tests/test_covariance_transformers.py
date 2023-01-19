import numpy as np
import pandas as pd
import pytest
from coffeine.covariance_transformers import Riemann

n_subjects = 10
n_channels = 4


def toy_data(data_frame):
    X_cov = np.random.randn(
        n_subjects, n_channels, n_channels)
    for sub in range(n_subjects):
        X_cov[sub] = X_cov[sub] @ X_cov[sub].T
    if data_frame:
        X_cov = pd.DataFrame({'cov': list(X_cov)})
    return X_cov


@pytest.mark.parametrize('data_frame', [True, False])
def test_Riemann(data_frame):
    X_cov = toy_data(data_frame)
    transformer = Riemann(return_data_frame=False)
    Xt_cov = transformer.fit_transform(X_cov)
    assert Xt_cov.shape == (n_subjects, n_channels*(n_channels + 1) / 2)
