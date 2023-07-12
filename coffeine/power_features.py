from typing import Union
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from pyriemann.estimation import CospCovariances
import mne
from mne.io import BaseRaw
from mne.epochs import BaseEpochs


def _compute_covs_raw(raw, clean_events, frequency_bands, duration, method):
    covs = list()
    for _, fb in frequency_bands.items():
        rf = raw.copy().load_data().filter(fb[0], fb[1])
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=None)
        cov = mne.compute_covariance(ec, method=method, rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_covs_epochs(epochs, frequency_bands, method):
    covs = list()
    for _, fb in frequency_bands.items():
        ec = epochs.copy().load_data().filter(fb[0], fb[1])
        cov = mne.compute_covariance(ec, method=method, rank=None)
        covs.append(cov.data)
    return np.array(covs)


def _compute_cross_frequency_covs(epochs, frequency_bands, method):
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
    cov = mne.compute_covariance(epochs_final, method=method, rank=None)
    corr = np.corrcoef(
        epochs_final.get_data().transpose((1, 0, 2)).reshape(n_chan, -1))
    return cov.data, corr


def _compute_cospectral_covs(epochs, n_fft, n_overlap, fmin, fmax, fs,
                             method):
    X = epochs.get_data()
    cospectral_covs = CospCovariances(window=n_fft, overlap=n_overlap/n_fft,
                                      fmin=fmin, fmax=fmax, fs=fs)
    return cospectral_covs.transform(X).mean(axis=0).transpose((2, 0, 1))


def get_frequency_bands(collection: str = 'ipeg',
                        subset: Union[list, tuple, None] = None) -> dict:
    """Get pre-specified frequency bands based on the literature.

    Next to sets of bands for defining filterbank models, the aggregate
    defined in the corresponding literature are provided.

    .. note::
        The HCP-MEG[1] frequency band was historically based on the
        documentation of the MEG analysis from the HCP-500 MEG2 release:
        https://wiki.humanconnectome.org/display/PublicData/MEG+Data+FAQ

        As frequencies below 1.5Hz were omitted the work presented in [2,3]
        also defined a 'low' band (0.1 - 1.5Hz) while retaining the the other
        frequencies.

    .. note::
        The IPEG frequency bands were developed in [4].

    .. note::
        Additional band definitions can be added as per (pull) request.

    Parameters
    ----------
    collection : {'ipeg', 'ipeg_aggregated', 'hcp', 'hcp_aggregated'}
        The set of frequency bands. Defaults to 'hcp'.
    subset : list-like
        A selection of valid keys to return a subset of frequency
        bands from a collection.

    Returns
    -------
    frequency_bands : dict
        The band definitions.

    References
    ----------
    [1] Larson-Prior, L. J., R. Oostenveld, S. Della Penna, G. Michalareas,
        F. Prior, A. Babajani-Feremi, J-M Schoffelen, et al. 2013.
        “Adding Dynamics to the Human Connectome Project with MEG.”
        NeuroImage 80 (October): 190–201.
    [2] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
        Predictive regression modeling with MEG/EEG: from source power
        to signals and cognitive states.
        *NeuroImage*, page 116893,2020. ISSN 1053-8119.
        https://doi.org/10.1016/j.neuroimage.2020.116893
    [3] D. A. Engemann, O. Kozynets, D. Sabbagh, G. Lemaître, G. Varoquaux,
        F. Liem, and A. Gramfort Combining magnetoencephalography with
        magnetic resonance imaging enhances learning of surrogate-biomarkers.
        eLife, 9:e54055, 2020 <https://elifesciences.org/articles/54055>
    [4] Jobert, M., Wilson, F.J., Ruigt, G.S., Brunovsky, M., Prichep,
        L.S., Drinkenburg, W.H. and IPEG Pharmaco-EEG Guideline Committee,
        2012. Guidelines for the recording and evaluation of pharmaco-EEG data
        in man: the International Pharmaco-EEG Society (IPEG).
        Neuropsychobiology, 66(4), pp.201-220.
    """
    frequency_bands = dict()
    if collection == 'ipeg':
        frequency_bands.update({
            "delta": (1.5, 6.0),
            "theta": (6.0, 8.5),
            "alpha1": (8.5, 10.5),
            "alpha2": (10.5, 12.5),
            "beta1": (12.5, 18.5),
            "beta2": (18.5, 21.0),
            "beta3": (21.0, 30.0),
            "gamma": (30.0, 40.0),
        })  # total: 1.5-30; dominant: 6-12.5
    if collection == 'ipeg_aggregated':
        frequency_bands.update({
            'total': (1.5, 30),
            'dominant': (6, 12.5)
        })
    elif collection == 'hcp':
        # https://www.humanconnectome.org/storage/app/media/documentation/
        # s500/hcps500meg2releasereferencemanual.pdf
        frequency_bands.update({
            'low': (0.1, 1.5),  # added later in [2,3].
            'delta': (1.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 15.0),
            'beta_low': (15.0, 26.0),
            'beta_high': (26.0, 35.0),
            'gamma_low': (35.0, 50.0),
            'gamma_mid': (50.0, 76.0),
            'gamma_high': (76.0, 120.0)
        })
    elif collection == 'hcp_aggregated':
        frequency_bands.update({
            'wide_band': (1.5, 150.0)
        })
    if subset is not None:
        frequency_bands = {
            band: freqs for band, freqs in frequency_bands.items()
            if band in subset
        }
    return frequency_bands


def make_coffeine_df(C: np.ndarray,
                     names: Union[dict, list, tuple, None] = None):
    """Put covariances in coffeine Data Frame.

    Parameters
    ----------
    C : np.ndarray, shape(n_obs, n_frequencies, n_channels, n_channels)
        A 2D collection of symmetric matrices. First dimension: samples.
        Second dimension: batches within observations (e.g. frequencies).
    names : dict or list-like, defaults to None
        A descriptor for the second dimension of `C`. It is used to make
        the columns of the coffeine Data Frame

    Returns
    -------
    C_df : pd.DataFrame
        The DataFrame of object type with lists of covariances accessible
        as columns.
    """
    assert C.ndim == 4
    assert C.shape[2] == C.shape[3]

    names_ = None
    if names is None:
        names_ = [f'c{cc}' for cc in range(C.shape[1])]
    else:
        names_ = names

    C_df = pd.DataFrame(
        {name: list(C[:, ii]) for ii, name in enumerate(names_)}
    )
    return C_df


def compute_coffeine(
        inst: Union[mne.io.BaseRaw, mne.BaseEpochs],
        frequencies: Union[str, tuple, dict] = 'ipeg',
        methods_params: Union[None, dict] = None
        ) -> pd.DataFrame:
    """Compute & spectral features as SPD matrices in a Data Frame.

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs or list-like
        The MNE instance containing raw signals from which to compute
        the features. If list-like, expected to contain MNE-Instances.
    frequencies : str | dict
        The frequency parameter. Either the name of a collection supported
        by `get_frequency_bands`or a dictionary of frequency names and ranges.
    methods_params : dict
        The methods paramaters used in the down-stream function for feature
        computation.

    Returns
    -------
    C_df : pd.DataFrame
        The coffeine DataFrame with columns filled with object arrays of
        covariances.
    """
    instance_list = list()
    if isinstance(inst, mne.io.BaseRaw):
        instance_list.append(inst)
    elif isinstance(inst, mne.BaseEpochs):
        if len(inst) == 1:
            instance_list.append(inst)
        elif len(inst) > 1:
            for ii in range(len(inst)):
                instance_list.append(inst[ii])
    elif isinstance(inst, list):
        instance_list.extend(inst)
    else:
        raise ValueError('Unexpected value for instance.')
    assert len(instance_list) >= 1
    frequencies_ = None
    if frequencies in ('ipeg', 'hcp'):
        frequencies_ = get_frequency_bands(collection=frequencies)
    elif isinstance(frequencies, tuple) and frequencies[0] in ('ipeg', 'hcp'):
        frequencies_ = get_frequency_bands(
            collection=frequencies[0], subset=frequencies[1]
        )
    elif isinstance(frequencies, dict):
        frequencies_ = frequencies
    else:
        raise NotImplementedError(
            'Currently, only collection names or fully-spelled band ranges are'
            ' supported as frequency definitions.'
        )

    freq_values = sum([list(v) for v in frequencies_.values()], [])
    methods_params_fb_bands_ = dict(
        features=('covs',), n_fft=1024, n_overlap=512,
        cov_method='oas', fs=instance_list[0].info['sfreq'],
        frequency_bands=frequencies_,
        fmin=min(freq_values), fmax=max(freq_values)
    )
    if methods_params is not None:
        methods_params_fb_bands_.update(methods_params)

    C = list()
    for ii, this_inst in enumerate(instance_list):
        features, feature_info = compute_features(
            this_inst, **methods_params_fb_bands_
        )
        C.append(features['covs'])
    C = np.array(C)
    C_df = make_coffeine_df(C=C, names=frequencies_)
    return C_df, feature_info


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
        cov_method='oas',
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
    cov_method : str (default 'oas')
        The covariance estimator to be used. Ignored for feature types not
        not related to covariances. Must be a method accepted by MNE's
        covariance functions.
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
                                     duration, method=cov_method)
            computed_features['covs'] = covs

    elif isinstance(inst, BaseEpochs):
        epochs_clean = clean_func(inst)
        if 'covs' in features:
            covs = _compute_covs_epochs(epochs_clean, frequency_bands_,
                                        method=cov_method)
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
            epochs_clean, frequency_bands_, method=cov_method)
        computed_features['cross_frequency_covs'] = cross_frequency_covs
        computed_features['cross_frequency_corrs'] = cross_frequency_corrs

    if 'cospectral_covs' in features:
        cospectral_covs = _compute_cospectral_covs(epochs_clean, n_fft,
                                                   n_overlap,
                                                   fmin, fmax, fs,
                                                   method=cov_method)
        computed_features['cospectral_covs'] = cospectral_covs

    return computed_features, res
