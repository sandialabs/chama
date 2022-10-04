.. raw:: latex

    \newpage

Installation
======================================

Chama requires Python (tested on 3.7, 3.8, and 3.9) along with several Python package dependencies.  
Information on installing and using Python can be found at 
https://www.python.org/.  
Python distributions, such as Anaconda, are recommended to manage the Python interface.  

To install the latest stable version of Chama using pip::

	pip install chama

To install the development branch of Chama from source using git::

	git clone https://github.com/sandialabs/chama
	cd chama
	python setup.py install

Developers should build Chama using the setup.py 'develop' option.

Dependencies
--------------
Required Python package dependencies include:

* Pyomo [HLWW12]_: Used to formulate optimization problems and call solvers, 
  https://github.com/pyomo. 
* Pandas [Mcki13]_: Used to analyze and store dataframes, 
  http://pandas.pydata.org.
* Numpy [VaCV11]_: Used to analyze large, multi-dimensional arrays and matrices, 
  http://www.numpy.org.
* Scipy [VaCV11]_: Used to support efficient routines for numerical analysis, 
  http://www.scipy.org.
  
Optional Python package dependencies include:

* Matplotlib [Hunt07]_: Used to produce graphics, 
  http://matplotlib.org.
* nose: Used to run software tests, http://nose.readthedocs.io.

Required Pyomo supported MIP solver:

* In addition to the Python package dependencies, a Pyomo supported MIP solver is required to solve the 
  optimization problems formulated in Chama. Examples of solvers that meet
  this requirement include GLPK [Makh10]_, Gurobi [GUROBI]_, and CPLEX [CPLEX]_.
* GLPK can be installed through conda-forge, `conda install -c conda-forge glpk`