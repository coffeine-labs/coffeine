import numpy as np
import pandas as pd
import pytest
from coffeine import (make_filter_bank_regressor,
                      make_filter_bank_classifier)

frequency_bands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
n_subjects = 10
n_channels = 4
n_frequency_bands = len(frequency_bands)


@pytest.fixture
def toy_data():
    X_cov = np.random.randn(
        n_subjects, n_frequency_bands, n_channels, n_channels)
    for sub in range(n_subjects):
        for fb in range(n_frequency_bands):
            X_cov[sub, fb] = X_cov[sub, fb] @ X_cov[sub, fb].T
    df = pd.DataFrame({band: list(X_cov[:, ii])
                       for ii, band in enumerate(frequency_bands)})
    df['drug'] = np.random.randint(2, size=n_subjects)
    rng = np.random.RandomState(2021)
    y = rng.randn(len(df))
    return df, y


def test_pipelines(toy_data):
    methods = (
        'riemann',
        'lw_riemann',
        'diag',
        'log_diag',
        'random',
        'naive',
        'spoc',
        'riemann_wasserstein'
    )
    for method in methods:
        regressor = make_filter_bank_regressor(
            names=frequency_bands.keys(),
            method=method,
            projection_params=(dict(n_compo=2) if method in
                               ('random', 'riemann', 'spoc') else None),
            categorical_interaction="drug")
        X_df, y = toy_data
        regressor.fit(X_df, y)

    with pytest.raises(ValueError, match=r".* specified .*"):
        regressor = make_filter_bank_regressor(
            names=frequency_bands.keys(),
            method='deep neural ANOVA')

    regressor = make_filter_bank_classifier(
        names=frequency_bands.keys(),
        method='riemann',
        vectorization_params=dict(metric='riemann'),
        categorical_interaction="drug")
    y_bin = np.sign(y - np.mean(y))
    regressor.fit(X_df, y_bin)
