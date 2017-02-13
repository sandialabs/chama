.. raw:: latex

    \newpage

Installation
======================================

Chama requires Python (2.7, 3.4, or 3.5) along with several python package dependencies.  
Information on installing and using python can be found at 
https://www.python.org/.  
Python distributions, such as Anaconda, are recommended to manage the Python interface.  

To build Chama from source using git::

	git clone https://github.com/sandialabs/chama
	cd chama
	python setup.py install

Python package dependencies include:

* Pyomo [HLWW12]_: optimization modeling language and optimization capabilities, 
  https://github.com/pyomo. 
* Pandas [Mcki13]_: analyze and store databases, 
  http://pandas.pydata.org.
* Numpy [VaCV11]_: support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org.

Optional python packages include:

* Matplotlib [Hunt07]_: produce figures, 
  http://matplotlib.org.
  
	