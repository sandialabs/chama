Overview
================

The sensor design optimization methods used in Chama were originally 
developed by Sandia National Laboratories and the U.S. Environmental Protection Agency
for water utilities.  
The basic sensor placement optimization method included in Chama is based on methods in
the Threat Ensemble Vulnerability Assessment and Sensor Placement Optimization Tool (TEVA-SPOT) [USEPA12]_
and the Water Security Toolkit (WST) [USEPA15]_.  
These tools embed transport simulations using the water distribution network model, EPANET [Ross00]_.

Chama was developed to be a general purpose sensor design tool.  
Chama expands on previous research by allowing the user to optimize both the location and type of sensors in a monitoring system.
Furthermore, transport simulations can come from a wide range of sources, including (but not limited to):

* Air dispersion models (e.g. AERMOD, CALPUFF, FLACS CFD)
* Water distribution network models (e.g. EPANET)
* Ground water models (e.g. MODFLOW)
* Seismic models

Sensor design optimization is determined using signal timeseries data generated from transport simulations.  
The signal timeseries records the time and location (x,y,z) a signal was simulated.  
Since current software development is focused on designing sensor networks for airborne pollutants, the examples used in 
this user manual focus on signal timeseries where the signal is air quality concentration.  
Multiple scenarios are considered to account for uncertainty in the wind speed and direction as well 
as the pollutant emission rate and location.

Chama includes methods to run simple Gaussian plume and puff air dispersion models.  
To use other transport simulations, such as the examples listed above, results can be formatted into general 
formats and imported into Chama.
Chama uses Pandas DataFrames [Mcki13]_ to store simulation results.  
Pandas includes many functions to easily load data from a wide range of file formats. 

Chama includes the ability to:

* Define models for differnet types of sensor technology, including stationary and mobile sensors, point detectors and cameras
* Optimize sensor location and type using impact statistics
* Generate maps of the site that include concentration plumes and sensor layout
