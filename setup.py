#! /usr/bin/env python

import os
from setuptools import setup, find_packages

descr = """M/EEG power regression pipelines in Python"""

version = None
with open(os.path.join('meegpowreg', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


DISTNAME = 'meegpowreg'
DESCRIPTION = descr
MAINTAINER = 'David Sabbagh'
MAINTAINER_EMAIL = 'david@sabbagh.fr'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/DavidSabbagh/meegpowreg.git'
VERSION = version
URL = 'https://github.com/DavidSabbagh/meegpowreg'

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
              'numpy>=1.12',
              'scipy>=0.18.0',
              'matplotlib>=2.0.0',
              'pandas',
              'pyriemann',
              'scikit-learn==0.23.2',  # due to pyriemann
              # 'scikit-learn>=0.23',
              'mne>=0.20'
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
