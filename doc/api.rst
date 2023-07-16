.. _api_ref:

=============
API reference
=============


Composing modeling pipelines
----------------------------
.. _pipeline_api:
.. currentmodule:: coffeine.pipelines

.. autosummary::
    :toctree: generated/
    :template: function.rst
    
    make_filter_bank_transformer
    make_filter_bank_regressor
    make_filter_bank_classifier


.. autosummary::
    :toctree: generated/
    :template: class.rst
    
    GaussianKernel
    KernelSum
    

Computing Power-Spectral Features
---------------------------------
.. _features_api:
.. currentmodule:: coffeine.power_features

.. autosummary::
    :toctree: generated/
    :template: function.rst

    get_frequency_bands
    make_coffeine_df
    compute_coffeine
    compute_features


Covariance Transformers
-----------------------
.. _covariance_transformer_api:
.. currentmodule:: coffeine.covariance_transformers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    NaiveVec
    Diag
    LogDiag
    Riemann
    RiemannSnp
    Snp
    ExpandFeatures
    

Spatiel Filters
---------------
.. _spatial_filters_api:
.. currentmodule:: coffeine.spatial_filters

.. autosummary::
    :toctree: generated/
    :template: class.rst
    
    ProjIdentitySpace
    ProjCommonSpace
    ProjLWSpace
    ProjRandomSpace
    ProjSPoCSpace

