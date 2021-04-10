import os

import numpy as np
import matplotlib.pyplot as plt
import mne

from covpredict.power_features import compute_features
from covpredict.spfiltering import ProjSPoCSpace


def test_spfiltering():
    n_compo = 'full'
    scale = 'auto'
    reg = 0
    shrink = 0.55

    data_path = mne.datasets.sample.data_path()
    data_dir = os.path.join(data_path, 'MEG', 'sample')
    raw_fname = os.path.join(data_dir, 'sample_audvis_raw.fif')
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw = raw.copy().crop(0, 100).pick(['mag']).pick(
        list(range(5))
    )
    raw.info.normalize_proj()
    info = raw.info

    fbands = {'alpha': (8.0, 15.0)}
    features, _ = compute_features(raw, fbands=fbands)
    cov = features['covs'][0]
    X = np.array([cov, cov])
    y = np.random.randint(100, size=2)

    spoc = ProjSPoCSpace(shrink=shrink, scale=scale, n_compo=n_compo, reg=reg)
    spoc.fit(X, y)

    n_compo = X.shape[1]
    fig, ax = plt.subplots(1, 2 * n_compo)
    spoc.plot_patterns(info=info, components=None, show=False,
                       name_format='', axes=ax[:n_compo], colorbar=False)
    spoc.plot_filters(info=info, components=None, show=False,
                      name_format='', axes=ax[n_compo:], colorbar=False)
