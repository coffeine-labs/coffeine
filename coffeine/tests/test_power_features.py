import os
import pytest
import mne


from coffeine.power_features import compute_features, get_frequency_bands

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


def test_get_frequency_bands():
    fbands_ipeg = get_frequency_bands(collection='ipeg')
    assert fbands_ipeg == {
        'delta': (1.5, 6.0),
        'theta': (6.0, 8.5),
        'alpha1': (8.5, 10.5),
        'alpha2': (10.5, 12.5),
        'beta1': (12.5, 18.5),
        'beta2': (18.5, 21.0),
        'beta3': (21.0, 30.0),
        'gamma': (30.0, 40.0)
    }
    fbands_ipeg_agg = get_frequency_bands(
        collection='ipeg_aggregated')
    assert fbands_ipeg_agg == {
        'total': (1.5, 30),
        'dominant': (6, 12.5)
    }
    fbands_hcp = get_frequency_bands(collection='hcp')
    assert fbands_hcp == {
        'low': (0.1, 1.5),
        'delta': (1.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 15.0),
        'beta_low': (15.0, 26.0),
        'beta_high': (26.0, 35.0),
        'gamma_low': (35.0, 50.0),
        'gamma_mid': (50.0, 76.0),
        'gamma_high': (76.0, 120.0)
    }
    fbands_hcp_agg = get_frequency_bands(
        collection='hcp_aggregated')
    assert fbands_hcp_agg == {'wide_band': (1.5, 150)}

    fbands_ipeg_subset = get_frequency_bands(
        collection='ipeg', subset=['alpha1', 'alpha2'])
    assert fbands_ipeg_subset == {
        'alpha1': (8.5, 10.5),
        'alpha2': (10.5, 12.5)
    }
    with pytest.raises(KeyError, match="alpha3"):
        fbands_ipeg_subset = get_frequency_bands(
            collection='ipeg', subset=['alpha1', 'alpha3'])
    with pytest.raises(ValueError,
                       match='"Hans Berger" is not a valid collection'):
        fbands_ipeg_subset = get_frequency_bands(
            collection='Hans Berger')
