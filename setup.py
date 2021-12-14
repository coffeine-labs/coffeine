#! /usr/bin/env python

import os
from setuptools import setup, find_packages

descr = """M/EEG power regression pipelines in Python"""

version = None
with open(os.path.join('coffeine', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'coffeine'
DESCRIPTION = descr
MAINTAINER = 'Denis Engemann'
MAINTAINER_EMAIL = 'denis.engemann@gmail.com'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/coffeine-labs/coffeine.git'
VERSION = version
URL = 'https://github.com/coffeine-labs/coffeine'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          keywords="",
          license=LICENSE,
          version=VERSION,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          python_requires=">=3",
          install_requires=[
              'numpy>=1.18.1',
              'scipy>=1.4.1',
              'matplotlib>=2.0.0',
              'pandas>=1.0.0',
              'pyriemann>=0.2.7',
              'scikit-learn>=0.24',
              'mne>=0.24'
          ],
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=find_packages(),
          )
