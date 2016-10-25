Installation
======================================

Chama requires Python 2.7 along with several python package dependencies.  
Information on installing and using python can be found at 
https://www.python.org/.  
Python distributions, such as Anaconda, are recommended to manage the Python interface.  

To build Chama from source using git::

	git clone https://github.com/sandialabs/chama
	cd chama
	python setup.py install

Python package dependencies include:

* Pyomo [Hart2012]_: optimization modeling language and optimization capabilities, 
  https://github.com/pyomo. 
* Pandas [McKinney2013]_: analyze and store databases, 
  http://pandas.pydata.org.
* Numpy [vanderWalt2011]_: support large, multi-dimensional arrays and matrices, 
  http://www.numpy.org.

Optional python packages include:

* Matplotlib [Hunter2007]_: produce figures, 
  http://matplotlib.org.
* Folium [Story2016]_: produce maps,
  https://github.com/python-visualization/folium.
  
	