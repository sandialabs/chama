Overview
================

The sensor placement optimization methods used in Chama were originally 
developed by Sandia National Laboratories and the U.S. Environmental Protection Agency
for water utilities.  
The basic sensor placement optimization method included in Chama is based on methods in
the Threat Ensemble Vulnerability Assessment and Sensor Placement Optimization Tool (TEVA-SPOT) [USEPA2012]_
and the Water Security Toolkit (WST) [USEPA2014]_.  
These tools embed transport simulations using the water distribution network model, EPANET [Rossman2000]_.

Chama was developed to be a general purpose sensor design tool.  
Chama expands on previous research by allowing the user to optimize both the location and type of sensors in a monitoring system.
Furthermore, transport simulations can come from a wide range of sources, including (but not limited to):

* Air dispersion models (e.g. AERMOD, CALPUFF)
* Water distribution network models (e.g. EPANET)
* Ground water models (e.g. MODFLOW)
* Seismic models

Chama includes methods to run simple Gaussian Plume air dispersion models.  
All other transport simulations are run using third-party software and results are 
imported into Chama.  Chama uses 
Pandas DataFrames [McKinney2013]_ to store simulation results.  
Pandas includes many functions to easily load data from a wide range of file formats. 

Chama includes the ability to:

* Define a wide range of sensor technology, including stationary and mobile sensors, point detectors and cameras
* Optimize sensor location and type using coverage or impact statistics
* Generate maps of the site that include concentration plumes and sensor layout
