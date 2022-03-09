import os
import pytest
import mne


from coffeine.power_features import compute_features

data_path = mne.datasets.sample.data_path()
data_dir = os.path.join(data_path, 'MEG', 'sample')
raw_fname = os.path.join(data_dir, 'sample_audvis_raw.fif')

frequency_bands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
frequency_bands2 = {'theta': (4.0, 8.0), 'beta': (15.0, 30.0)}


def test_compute_features_raw():
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw = raw.copy().crop(0, 200).pick(
        [0, 1, 330, 331, 332]  # take some MEG and EEG
    )
    raw.info.normalize_proj()
    computed_features, res = compute_features(
        raw, features=['psds', 'covs', 'cross_frequency_covs',
                       'cross_frequency_corrs', 'cospectral_covs'],
        frequency_bands=frequency_bands)
    n_channels = len(raw.ch_names)
    n_freqs = len(res['freqs'])
    n_fb = len(frequency_bands)
    assert (
        set(computed_features.keys()) ==
        {'psds', 'covs', 'cross_frequency_covs',
         'cross_frequency_corrs', 'cospectral_covs'}
    )
    assert computed_features['psds'].shape == (n_channels, n_freqs)
    assert computed_features['covs'].shape == (n_fb, n_channels, n_channels)
    assert (computed_features['cross_frequency_covs'].shape ==
            (n_fb * n_channels, n_fb * n_channels))
    assert (computed_features['cross_frequency_corrs'].shape ==
            (n_fb * n_channels, n_fb * n_channels))
    assert (computed_features['cospectral_covs'].shape[1:] ==
            (n_channels, n_channels))

    with pytest.raises(ValueError, match=r".* specified .*"):
        computed_features, res = compute_features(
            raw, features='covs haha',
            frequency_bands=frequency_bands)


def test_compute_features_epochs():
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw = raw.copy().crop(0, 200).pick(
        [0, 1, 330, 331, 332]  # take some MEG and EEG
    )
    raw.info.normalize_proj()
    events = mne.make_fixed_length_events(raw, id=3000,
                                          start=0,
                                          duration=10.,
                                          stop=raw.times[-1] - 60.)
    epochs = mne.Epochs(raw, events, event_id=3000, tmin=0, tmax=60.,
                        proj=True, baseline=None, reject=None,
                        preload=True, decim=1)
    computed_features, res = compute_features(
        epochs, features=['psds', 'covs', 'cross_frequency_covs',
                          'cross_frequency_corrs', 'cospectral_covs'],
        frequency_bands=frequency_bands)
    n_channels = len(raw.ch_names)
    n_freqs = len(res['freqs'])
    n_fb = len(frequency_bands)
    assert set(computed_features.keys()) == {'psds', 'covs',
                                             'cross_frequency_covs',
                                             'cross_frequency_corrs',
                                             'cospectral_covs'}

    assert computed_features['psds'].shape == (n_channels, n_freqs)
    assert computed_features['covs'].shape == (n_fb, n_channels, n_channels)
    assert (computed_features['cross_frequency_covs'].shape ==
            (n_fb * n_channels, n_fb * n_channels))
    assert (computed_features['cross_frequency_corrs'].shape ==
            (n_fb * n_channels, n_fb * n_channels))
    assert (computed_features['cospectral_covs'].shape[1:] ==
            (n_channels, n_channels))


@pytest.mark.parametrize('frequency_bands',
                         [None, frequency_bands, frequency_bands2])
def test_compute_features_covs_freq_band_defaults(frequency_bands):
    raw = mne.io.read_raw_fif(raw_fname, verbose=False)
    raw = raw.copy().crop(0, 200).pick(
        [0, 1, 330, 331, 332]  # take some MEG and EEG
    )
    raw.info.normalize_proj()
    computed_features, _ = compute_features(
        raw, features=['covs'], frequency_bands=frequency_bands)

    n_bands = computed_features['covs'].shape[0]
    assert n_bands == 1 if frequency_bands is None \
        else n_bands == len(frequency_bands)
