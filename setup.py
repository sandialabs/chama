from setuptools import setup, find_packages
from distutils.core import Extension
from os import path

DISTNAME = 'chama'
VERSION = '0.1.2'
PACKAGES = ['chama']
EXTENSIONS = []
DESCRIPTION = 'Sensor Placement Optimization.'
AUTHOR = 'Chama developers'
MAINTAINER_EMAIL = 'kaklise@sandia.gov'
LICENSE = 'Revised BSD'
URL = 'https://github.com/sandialabs/chama'

# use README file as the long description
file_dir = path.abspath(path.dirname(__file__))
with open(path.join(file_dir, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()
	
setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': [],
    'scripts': [],
    'include_package_data': True
}

setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=EXTENSIONS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)
