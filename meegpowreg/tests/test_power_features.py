import os
import mne

from covpredict.power_features import compute_features

data_path = mne.datasets.sample.data_path()
data_dir = os.path.join(data_path, 'MEG', 'sample')
raw_fname = os.path.join(data_dir, 'sample_audvis_raw.fif')

fbands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}


def test_compute_features():
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw = raw.copy().crop(0, 200).pick(
        list(range(2)) + list(range(330, 333))  # take some MEG and EEG
    )
    features, res = compute_features(raw, fbands=fbands)
    n_channels = len(raw.ch_names)
    n_freqs = len(features['freqs'])
    n_fb = len(fbands)
    assert (
        set(features.keys()) ==
        set(['psds', 'freqs', 'covs', 'xfreqcovs', 'xfreqcorrs', 'cospcovs'])
    )
    assert features['psds'].shape == (n_channels, n_freqs)
    assert features['covs'].shape == (n_fb, n_channels, n_channels)
    assert features['xfreqcovs'].shape == (n_fb * n_channels,
                                           n_fb * n_channels)
    assert features['xfreqcorrs'].shape == (n_fb * n_channels,
                                            n_fb * n_channels)
    assert features['cospcovs'].shape[1:] == (n_channels, n_channels)
