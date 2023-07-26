# Covariance Data Frames for Predictive M/EEG Pipelines

![Build](https://github.com/coffeine-labs/coffeine/workflows/tests/badge.svg)
<!-- ![Codecov](https://codecov.io/gh/coffeine-labs/coffeine/branch/main/graph/badge.svg) -->

Coffeine is designed for building biomedical prediction models from M/EEG signals. The library provides a high-level interface facilitating the use of M/EEG covariance matrix as representation of the signal. The methods implemented here make use of tools and concepts implemented in [PyRiemann](https://pyriemann.readthedocs.io/). The API is fully compatible with [scikit-learn](https://scikit-learn.org/) and naturally integrates with [MNE](https://mne.tools). 


```python
import mne
from coffeine import compute_coffeine, make_filter_bank_regressor

# load EEG data from linguistic experiment
eeg_fname = mne.datasets.kiloword.data_path() / "kword_metadata-epo.fif"
epochs = mne.read_epochs(eeg_fname)[:50]  # 50 samples

# compute covariances in different frequency bands 
X_df, feature_info = compute_coffeine(  # (defined by IPEG consortium)
    epochs, frequencies=('ipeg', ('delta', 'theta', 'alpha1'))
)  # ... and put results in a pandas DataFrame.
y = epochs.metadata["WordFrequency"]  # regression target

# compose a pipeline
model = make_filter_bank_regressor(method='riemann', names=X_df.columns)
model.fit(X_df, y)
```
<img width="1424" alt="image" src="https://github.com/coffeine-labs/coffeine/assets/1908618/161b997c-a51b-4885-9775-a1e5b84e10f9">

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
for ii, name in enumerate(('delta', 'theta', 'alpha1')):
    axes[ii].matshow(X_df[name].mean(), cmap='PuOr')
    axes[ii].set_title(name)
```
    
<img width="1108" alt="image" src="https://github.com/coffeine-labs/coffeine/assets/1908618/0d8e3882-177b-4dc2-9f60-343870713a82">

## Background

For this purpose, `coffeine` uses DataFrames to handle multiple covariance matrices alongside scalar features. Vectorization and model composition functions are provided that handle composition of valid [scikit-learn](https://scikit-learn.org/) modeling pipelines from covariances alongside other types of features as inputs.

The filter-bank pipelines (e.g. across multiple frequency bands or conditions) can the thought of as follows:

![](https://user-images.githubusercontent.com/1908618/115611659-a6d5ab80-a2ea-11eb-935c-006cad4fc8e5.png)
**M/EEG covariance-based modeling pipeline from [Sabbagh et al. 2020, NeuroImage](https://doi.org/10.1016/j.neuroimage.2020.116893https://doi.org/10.1016/j.neuroimage.2020.116893)**

After preprocessing, covariance matrices can be ___projected___ to a subspace by spatial filtering to mitigate field spread and deal with rank deficient signals.
Subsequently, ___vectorization___ is performed to extract column features from the variance, covariance or both.
Every path combnining different lines in the graph describes one particular prediction model.
The Riemannian embedding is special in mitigating field spread and providing vectorization in 1 step.
It can be combined with dimensionality reduction in the projection step to deal with rank deficiency.
Finally, a statistical learning algorithm can be applied.

The representation, projection and vectorization steps are separately done for each frequency band (or condition).

## Installation of Python package

<!-- To install the package, simply do: -->
<!--  -->
<!--   `$ pip install coffeine` -->

You can clone this library, and then do:

  `$ pip install -e .`

Everything worked if the following command do not return any error:

  `$ python -c 'import coffeine'`


## Citation

When publishing research using coffeine, please cite our core paper.

```
@article{sabbagh2020predictive,
  title={Predictive regression modeling with MEG/EEG: from source power to signals and cognitive states},
  author={Sabbagh, David and Ablin, Pierre and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Engemann, Denis A},
  journal={NeuroImage},
  volume={222},
  pages={116893},
  year={2020},
  publisher={Elsevier}
}
```

Please cite additional references highlighted in the documentation of specific functions and tutorials when using these functions and examples. 

Please also cite the upstream software this package is building on, in particular [PyRiemann](https://pyriemann.readthedocs.io/).
