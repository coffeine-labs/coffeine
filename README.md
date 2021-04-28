# Power regression pipelines for MEG/EEG signals

![Build](https://github.com/DavidSabbagh/meegpowreg/workflows/tests/badge.svg)
<!-- ![Codecov](https://codecov.io/gh/DavidSabbagh/meegpowreg/branch/main/graph/badge.svg) -->

## Summary

This library implements the methods used in the following articles:

[1] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D. A. Engemann.
Predictive regression modeling with MEG/EEG: from source power to signals and cognitive states.
*NeuroImage*, page 116893,2020. ISSN 1053-8119.
<https://www.sciencedirect.com/science/article/pii/S1053811920303797>

[2] D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort,
and D. A. Engemann.
Manifold-regression to predict from MEG/EEG brain signals
without source modeling.
*NeurIPS* (Advances in Neural Information Processing Systems) 32.
<https://papers.nips.cc/paper/8952-manifold-regression-to-predict-from-megeeg-brain-signals-without-source-modeling>

[3] D. A. Engemann, O. Kozynets, D. Sabbagh, G. Lemaître, G. Varoquaux, F. Liem, and A. Gramfort
Combining magnetoencephalography with magnetic resonance imaging enhances learning of surrogate-biomarkers.
*eLife*, 9:e54055, 2020
<https://elifesciences.org/articles/54055>

The filter bank pipelines can the thought of as follows:

<img width="1380" alt="meeg_pipelines" src="https://user-images.githubusercontent.com/1908618/115611659-a6d5ab80-a2ea-11eb-935c-006cad4fc8e5.png">

After preprocessing, covariance matrices can be projected to mitigate field spread and deal with rank deficient signals.
Subsequently, vectorization is performed to extract column features from the variance, covariance or both.
The Riemannian embedding is special in mititgating field spread and providing vectorization in 1 step.
It can be combined with dimensionality reduction in the projection step to deal with rank deficinency.
Finally, a statistical learning algorithm is applied.

## Installation of Python package

<!-- To install the package, simply do: -->
<!--  -->
<!--   `$ pip install meegpowreg` -->

You can clone this library, and then do:

  `$ pip install -e .`

Everything worked if the following command do not return any error:

  `$ python -c 'import meegpowreg'`

## Use with Python

### compute_features

Compute power features from raw M/EEG data:

- The power spectral density
- The spectral covariance matrices
- The cospectral covariance matrices
- The cross-frequency covariance matrices
- The cross-frequency correlation matrices

The matrices are of shape (n_frequency_bands, n_channels, n_channels)

Use case example:

```python
import os
import mne

from meegpowreg import compute_features

data_path = mne.datasets.sample.data_path()
data_dir = os.path.join(data_path, 'MEG', 'sample')
raw_fname = os.path.join(data_dir, 'sample_audvis_raw.fif')

raw = mne.io.read_raw_fif(raw_fname, verbose=False)
# pick some MEG and EEG channels after cropping
raw = raw.copy().crop(0, 200).pick([0, 1, 330, 331, 332])

fbands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}

features, _ = compute_features(raw, fbands=fbands)
```

### make_filter_bank_models

The following models are implemented:

- riemann
- lw_riemann
- diag
- logdiag
- random
- naive
- spoc
- riemann_wass
- dummy

Use case example:

```python
import numpy as np
import pandas as pd
from meegpowreg import make_filter_bank_regressor

freq_bands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
n_freq_bands = len(freq_bands)
n_subjects = 10
n_channels = 4

# Make toy data
X_cov = np.random.randn(n_subjects, n_freq_bands, n_channels, n_channels)
for sub in range(n_subjects):
    for fb in range(n_freq_bands):
        X_cov[sub, fb] = X_cov[sub, fb] @ X_cov[sub, fb].T
X_df = pd.DataFrame(
  {band: list(X_cov[:, ii]) for ii, band in enumerate(freq_bands)})
X_df['drug'] = np.random.randint(2, size=n_subjects)
y = np.random.randn(len(X_df))

# Models
fb_model = make_filter_bank_regressor(names=freq_bands.keys(),
                                      method='riemann')
fb_model.fit(X_df, y)
```

## Cite

If you use this code please cite:

  D. Sabbagh, P. Ablin, G. Varoquaux, A. Gramfort, and D.A. Engemann.
  Predictive regression modeling with MEG/EEG: from source power
  to signals and cognitive states.
  *NeuroImage*, page 116893,2020. ISSN 1053-8119.
  https://www.sciencedirect.com/science/article/pii/S1053811920303797
