.. raw:: latex

    \newpage

Installation
======================================

Chama requires Python (2.7, 3.4, 3.5, or 3.6) along with several Python package dependencies.  
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

* Pyomo [HLWW12]_: formulate optimization methods, 
  https://github.com/pyomo. 
* Pandas [Mcki13]_: analyze and store databases, 
  http://pandas.pydata.org.
* Numpy [VaCV11]_: support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org.
* Scipy [VaCV11]_: support efficient routines for numerical analysis, 
  http://www.scipy.org.
  
Optional Python package dependencies include:

* Matplotlib [Hunt07]_: produce graphics, 
  http://matplotlib.org.
* nose: run software tests, http://nose.readthedocs.io.

In addition, **a Pyomo supported MIP solver is required** to solve the
optimization problems formulated in Chama. Examples of solvers that meet
this requirement include GLPK [Makh10]_, Gurobi [GUROBI]_, and CPLEX [CPLEX]_.