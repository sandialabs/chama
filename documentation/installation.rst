.. raw:: latex

    \newpage

Installation
======================================

Chama requires Python (2.7, 3.4, or 3.5) along with several python package dependencies.  
Information on installing and using python can be found at 
https://www.python.org/.  
Python distributions, such as Anaconda, are recommended to manage the Python interface.  

To install the latest stable version of Chama using pip::

	pip install chama

To build the development branch of Chama from source using git::

	git clone https://github.com/sandialabs/chama
	cd chama
	python setup.py install

Developers should build Chama using the 'develop' option::

	git clone https://github.com/sandialabs/chama
	cd chama
	python setup.py develop

Dependencies
--------------
Python package dependencies include:

* Pyomo [HLWW12]_: optimization modeling language and optimization capabilities, 
  https://github.com/pyomo. 
* Pandas [Mcki13]_: analyze and store databases, 
  http://pandas.pydata.org.
* Numpy [VaCV11]_: support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org.

Optional python package dependencies include:

* Matplotlib [Hunt07]_: produce figures, 
  http://matplotlib.org.
 
Software QA
-------------------------------------

**GitHub:**
The software repository is hosted on GitHub at https://github.com/sandialabs/chama.  

**PyPI:**
The latest stable version hosted on PyPI at https://pypi.python.org/pypi/chama.

**Testing:**
Automated testing is run using TravisCI at https://travis-ci.org/sandialabs/chama.
Test coverage statistics are collected using Coveralls at https://coveralls.io/github/sandialabs/chama.
Tests can be run locally using nosetests::
  
	nosetests -v --with-coverage --cover-package=chama chama

**Documentation:**
Documentation is built using Read the Docs and hosted at https://chama.readthedocs.io.

**Contributing:**
Software developers are expected to follow standard practices to document and test new code. 
Pull requests will be reviewed by the core development team.
See https://github.com/sandialabs/chama/graphs/contributors for a list of contributors.

