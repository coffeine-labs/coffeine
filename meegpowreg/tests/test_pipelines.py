import numpy as np
import pandas as pd
import pytest
from covpredict.pipelines import make_pipelines

fbands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
n_sub = 10
n_ch = 4
n_fb = len(fbands)

pipelines = make_pipelines(
    fb_cols=fbands.keys(),
    expand=True
)

@pytest.fixture
def toy_data():
    Xcov = np.random.randn(n_sub, n_fb, n_ch, n_ch)
    for sub in range(n_sub):
        for fb in range(n_fb):
            Xcov[sub, fb] = Xcov[sub, fb] @ Xcov[sub, fb].T
    Xcov = list(Xcov.transpose((1, 0, 2, 3)))
    df = pd.DataFrame(dict(zip(list(fbands.keys()), map(list, Xcov))))
    df['drug'] = np.random.randint(2, size=n_sub)
    y = np.random.randn(len(df))
    return df, y


@pytest.mark.parametrize('pipeline_name', pipelines)
def test_pipelines(pipeline_name, toy_data):
    model = pipelines[pipeline_name]
    X_df, y = toy_data
    model.fit(X_df, y)
