import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.18'
PACKAGE_NAME = 'CIMLA'
AUTHOR = 'Payam Dibaeinia'
AUTHOR_EMAIL = 'dibaein2@illinois.edu'
URL = 'https://github.com/PayamDiba/CIMLA'

LICENSE = 'MIT License'
DESCRIPTION = 'Counterfactual Inference by Machine Learning and Attribution Models'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'dask==2021.10.0',
      'h5py==2.10.0',
      'numpy>1.22',
      'pandas==1.3.3',
      'scipy',
      'tensorflow==2.2.0',
      'dask-ml==1.9.0',
      'tables==3.6.*',
      'shap==0.39.0',
      'scikit-learn==0.24.2',
      'xgboost==1.5.0',
      'PyYAML==5.4.1',
      'protobuf==3.20',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      entry_points ={'console_scripts': [ 'cimla = CIMLA.main:main']},
      python_requires='==3.8.12',
      )
