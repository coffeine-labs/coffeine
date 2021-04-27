import numpy as np
import pandas as pd
import pytest
from meegpowreg.pipelines import make_filter_bank_model

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
    model = make_filter_bank_model(
        names=frequency_bands.keys(),
        pipeline='riemann',
        categorical_interaction="drug")
    X_df, y = toy_data
    model.fit(X_df, y)
