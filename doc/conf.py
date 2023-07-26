# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import pydata_sphinx_theme

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "coffeine")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'coffeine'
copyright = '2021-2023, coffeine contributors'
author = 'Denis A. Engemann'
release = '0.3dev'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'numpydoc',
    'sphinx_copybutton'
    # 'sphinx_gallery.gen_gallery'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
numpydoc_show_class_members = False 

# -- nbsphinx configuration --------------------------------------------------

nbsphinx_prolog = """
.. raw:: html

    <style>
        .nbinput .prompt,
        .nboutput .prompt {
            display: none;
        }
    </style>
"""

# -- Intersphinx configuration -----------------------------------------------
# copied from MNE

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/latest", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "pyriemann": ("https://pyriemann.readthedocs.io/en/latest", None),
    "mne": ("https://mne.tools/stable", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
    "mne-connectivity": ("https://mne.tools/mne-connectivity/stable", None),
    "mne-gui-addons": ("https://mne.tools/mne-gui-addons", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "altair": ("https://altair-viz.github.io/", None)
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Activate the theme.
html_theme = "pydata_sphinx_theme"
# html_context = {
#    "default_mode": "light",
# }

html_theme_options = {
  "header_links_before_dropdown": 4,
  "navbar_end": ["theme-switcher", "navbar-icon-links"],
  "external_links": [
        {
            "url": "https://pyriemann.readthedocs.io",
            "name": "PyRiemann",
        },
        {
            "url": "https://scikit-learn.org",
            "name": "scikit-learn",
        },
        {
            "url": "https://mne.tools",
            "name": "MNE-Python"
        }
    ],
    "icon_links": [
        {
            'name': "GitHub",
            'url': "https://github.com/coffeine-labs/coffeine",
            'icon': "fa-brands fa-square-github",
        }
    ],
    "icon_links_label": "External Links",  # for screen reader
    "use_edit_page_button": True,
    "navigation_with_keys": False
}

html_context = {
    "default_mode": "light",
    "github_user": "coffeine-labs",
    "github_repo": "coffeine",
    "github_version": "main",
    "doc_path": "doc"
}
