import numpy as np
from scipy.stats import trim_mean
from pyriemann.estimation import CospCovariances
import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs


def _compute_covs_raw(raw, clean_events, fbands, duration):
    covs = list()
    for _, fb in fbands.items():
        rf = raw.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=None)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_covs_epochs(epochs, fbands):
    covs = list()
    for _, fb in fbands.items():
        ec = epochs.copy().load_data().filter(fb[0], fb[1])
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_xfreq_covs(epochs, fbands):
    epochs_fbands = []
    for ii, (fbname, fb) in enumerate(fbands.items()):
        ef = epochs.copy().load_data().filter(fb[0], fb[1])
        for ch in ef.ch_names:
            ef.rename_channels({ch: ch+'_'+fbname})
        epochs_fbands.append(ef)

    epochs_final = epochs_fbands[0]
    for e in epochs_fbands[1:]:
        epochs_final.add_channels([e], force_update_info=True)
    n_chan = epochs_final.info['nchan']
    cov = mne.compute_covariance(epochs_final, method='oas', rank=None)
    corr = np.corrcoef(
        epochs_final.get_data().transpose((1, 0, 2)).reshape(n_chan, -1))
    return cov.data, corr


def _compute_cosp_covs(epochs, n_fft, n_overlap, fmin, fmax, fs):
    X = epochs.get_data()
    cosp_covs = CospCovariances(window=n_fft, overlap=n_overlap/n_fft,
                                fmin=fmin, fmax=fmax, fs=fs)
    return cosp_covs.transform(X).mean(axis=0).transpose((2, 0, 1))


def compute_features(
        inst,
        duration=60.,
        shift=10.,
        n_fft=512,
        n_overlap=256,
        fs=63.0,
        fmin=0,
        fmax=30,
        fbands={'alpha': (8.0, 12.0)},
        clean_func=lambda x: x,
        n_jobs=1):
    """Compute features from raw data or clean epochs.

    Parameters
    ----------
    inst : Raw object | Epochs object
        An instance of Raw or Epochs.
    duration : float
        The length of the epochs. If nothing is provided, defaults to 60.
    shift : float
        The duration to separate events by. If nothing is provided,
        defaults to 10.
    n_fft : int
        The length of FFT used for computing power spectral density (PSD)
        using Welch's method. If nothing is provided, defaults to 512.
    n_overlap : int
        The number of points of overlap between segments for PSD computation
        and for the estimation of cospectral covariance matrix.
        If nothing is provided, defaults to 256.
    fs : float
        The sampling frequency of the signal for the estimation of cospectral
        covariance matrix. If nothing is provided, defaults to 63.0.
    fmin : int
        The minimal frequency to be returned for the estimation of cospectral
        covariance matrix. If nothing is provided, defaults to 0.
    fmax : int
        The maximal frequency to be returned for the estimation of cospectral
        covariance matrix. If nothing is provided, defaults to 30.
    fbands : dict
        The frequency band with which inst is filtered. If nothing is provided,
        defaults to {'alpha': (8.0, 12.0)}.
    clean_func : lambda function
        If nothing is provided, defaults to lambda x: x.
    n_jobs : int
        If nothing is provided, defaults to 1.

    Returns
    -------
    features : dict
        The features extracted.
    res : dict
        The number of epochs, good epochs and clean epochs.
    """
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
        covs = _compute_covs_raw(inst, clean_events, fbands, duration)

    elif isinstance(inst, BaseEpochs):
        epochs_clean = clean_func(inst)
        covs = _compute_covs_epochs(epochs_clean, fbands)
    else:
        raise ValueError('Inst must be raw or epochs.')

    psds_clean, freqs = mne.time_frequency.psd_welch(
        epochs_clean, fmin=fmin, fmax=fmax, n_fft=n_fft, n_overlap=n_overlap,
        average='mean', picks=None)
    psds = trim_mean(psds_clean, 0.25, axis=0)

    xfreqcovs, xfreqcorrs = _compute_xfreq_covs(epochs_clean, fbands)
    cospcovs = _compute_cosp_covs(epochs_clean, n_fft, n_overlap,
                                  fmin, fmax, fs)

    features = {'psds': psds, 'freqs': freqs,
                'covs': covs,
                'xfreqcovs': xfreqcovs,
                'xfreqcorrs': xfreqcorrs,
                'cospcovs': cospcovs}

    res = dict(n_good_epochs=len(inst),
               n_clean_epochs=len(epochs_clean))
    if isinstance(inst, BaseRaw):
        res['n_epochs'] = len(events)
    else:
        res['n_epochs'] = len(inst.drop_log)
    return features, res
