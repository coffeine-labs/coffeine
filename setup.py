from setuptools import setup, find_packages

setup(
    name="covpredict",
    description="covpredict",
    keywords="",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=['numpy>=1.12', 'scipy>=0.18.0',
                      'matplotlib>=2.0.0',
                      'pandas',
                      'pyriemann',
                      'scikit-learn>=0.23', 'mne>=0.20']
)
