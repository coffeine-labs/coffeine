import numpy as np
from scipy.stats import trim_mean
from pyriemann.estimation import CospCovariances
import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs


def _compute_covs_raw(raw, clean_events, frequency_bands, duration):
    covs = list()
    for _, fb in frequency_bands.items():
        rf = raw.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=None)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_covs_epochs(epochs, frequency_bands):
    covs = list()
    for _, fb in frequency_bands.items():
        ec = epochs.copy().load_data().filter(fb[0], fb[1])
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_cross_frequency_covs(epochs, frequency_bands):
    epochs_frequency_bands = []
    for ii, (fbname, fb) in enumerate(frequency_bands.items()):
        ef = epochs.copy().load_data().filter(fb[0], fb[1])
        for ch in ef.ch_names:
            ef.rename_channels({ch: ch+'_'+fbname})
        epochs_frequency_bands.append(ef)

    epochs_final = epochs_frequency_bands[0]
    for e in epochs_frequency_bands[1:]:
        epochs_final.add_channels([e], force_update_info=True)
    n_chan = epochs_final.info['nchan']
    cov = mne.compute_covariance(epochs_final, method='oas', rank=None)
    corr = np.corrcoef(
        epochs_final.get_data().transpose((1, 0, 2)).reshape(n_chan, -1))
    return cov.data, corr


def _compute_cospectral_covs(epochs, n_fft, n_overlap, fmin, fmax, fs):
    X = epochs.get_data()
    cospectral_covs = CospCovariances(window=n_fft, overlap=n_overlap/n_fft,
                                      fmin=fmin, fmax=fmax, fs=fs)
    return cospectral_covs.transform(X).mean(axis=0).transpose((2, 0, 1))


def compute_features(
        inst,
        features=('psds', 'covs'),
        duration=60.,
        shift=10.,
        n_fft=512,
        n_overlap=256,
        fs=63.0,
        fmin=0,
        fmax=30,
        frequency_bands=None,
        clean_func=lambda x: x,
        n_jobs=1):
    """Compute features from raw data or clean epochs.

    Parameters
    ----------
    inst : Raw object | Epochs object
        An instance of Raw or Epochs.
    features : str | list of str
        The features to be computed. It can be 'psds', 'covs',
        'cross_frequency_covs', 'cross_frequency_corrs' or
        'cospectral_covs'. If nothing is provided, defaults to
        ('psds', 'covs').
    duration : float
        The length of the epochs. If nothing is provided, defaults to 60.
    shift : float
        The duration to separate events by (sliding shift of the epochs).
        If nothing is provided, defaults to 10.
    n_fft : int
        The length of FFT used for computing power spectral density (PSD)
        using Welch's method and the cospectral covariance.
        If nothing is provided, defaults to 512.
    n_overlap : int
        The number of points of overlap between segments for PSD computation
        and for the estimation of cospectral covariance matrix.
        If nothing is provided, defaults to 256.
    fs : float
        The sampling frequency of the signal for the estimation of cospectral
        covariance matrix. If nothing is provided, defaults to 63.0.
    fmin : int
        The minimal frequency to be returned for the estimation of cospectral
        covariance matrix and for PSD computation.
        If nothing is provided, defaults to 0.
    fmax : int
        The maximal frequency to be returned for the estimation of cospectral
        covariance matrix and for PSD computation.
        If nothing is provided, defaults to 30.
    frequency_bands : dict
        The frequency bands with which inst is filtered.
        If nothing is provided, defaults to {'alpha': (8.0, 12.0)}.
    clean_func : lambda function
        If nothing is provided, defaults to lambda x: x.
    n_jobs : int
        If nothing is provided, defaults to 1.

    Returns
    -------
    computed_features : dict
        The features extracted.
    res : dict
        The number of epochs, good epochs, clean epochs and frequencies.
    """
    features_ = (
        'psds', 'covs', 'cross_frequency_covs', 'cross_frequency_corrs',
        'cospectral_covs')

    frequency_bands_ = {'alpha': (8.0, 12.0)} \
        if frequency_bands is None else frequency_bands
    computed_features = {}

    if isinstance(inst, BaseRaw):
        events = mne.make_fixed_length_events(inst, id=3000,
                                              start=0,
                                              duration=shift,
                                              stop=inst.times[-1] - duration)
        epochs = mne.Epochs(inst, events, event_id=3000, tmin=0, tmax=duration,
                            proj=True, baseline=None, reject=None,
                            preload=True, decim=1)
        epochs_clean = clean_func(epochs)
        clean_events = events[epochs_clean.selection]
        if 'covs' in features:
            covs = _compute_covs_raw(inst, clean_events, frequency_bands_,
                                     duration)
            computed_features['covs'] = covs

    elif isinstance(inst, BaseEpochs):
        epochs_clean = clean_func(inst)
        if 'covs' in features:
            covs = _compute_covs_epochs(epochs_clean, frequency_bands_)
            computed_features['covs'] = covs
    else:
        raise ValueError('Inst must be raw or epochs.')

    res = dict(n_good_epochs=len(inst),
               n_clean_epochs=len(epochs_clean))
    if isinstance(inst, BaseRaw):
        res['n_epochs'] = len(events)
    else:
        res['n_epochs'] = len(inst.drop_log)

    if isinstance(features, str):
        features = [features]
    for feature in features:
        if feature not in features_:
            raise ValueError(
                f"The `features` ('{feature}') you specified is unknown.")

    if 'psds' in features:
        spectrum = epochs_clean.compute_psd(
                method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
                n_overlap=n_overlap, average='mean', picks=None)
        psds_clean = spectrum.get_data()
        psds = trim_mean(psds_clean, 0.25, axis=0)
        computed_features['psds'] = psds
        res['freqs'] = spectrum.freqs

    if ('cross_frequency_covs' in features or
            'cross_frequency_corrs' in features):
        (cross_frequency_covs,
            cross_frequency_corrs) = _compute_cross_frequency_covs(
            epochs_clean, frequency_bands_)
        computed_features['cross_frequency_covs'] = cross_frequency_covs
        computed_features['cross_frequency_corrs'] = cross_frequency_corrs

    if 'cospectral_covs' in features:
        cospectral_covs = _compute_cospectral_covs(epochs_clean, n_fft,
                                                   n_overlap,
                                                   fmin, fmax, fs)
        computed_features['cospectral_covs'] = cospectral_covs

    return computed_features, res
