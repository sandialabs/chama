.. raw:: latex
	
	\newpage

Overview
================

The sensor placement optimization methods used in Chama were originally 
developed by Sandia National Laboratories and the U.S. Environmental Protection Agency
for water utilities.  
The basic sensor placement optimization method included in Chama is based on methods in
the Threat Ensemble Vulnerability Assessment and Sensor Placement Optimization Tool (TEVA-SPOT) [USEPA12]_
and the Water Security Toolkit (WST) [USEPA15]_.  
These tools embed contaminant transport simulations, generated using the water distribution network model EPANET [Ross00]_, 
with the sensor placement optimization methods.

Chama was developed to be a general purpose sensor placement optimization software tool.  
Chama expands on previous methods by allowing the user to optimize both the location and type of sensors in a monitoring system.
Both stationary and mobile sensors can be defined for detection at a point or taking into account a field of view from a camera.  
Optical gas imaging models from the Fugitive Emissions Abatement Simulation Testbed (FEAST) [KeRB16]_ have been incorporated into Chama.
Furthermore, transport simulations can represent a wide range of applications, including (but not limited to):

* Air dispersion
* Transport in pipe networks
* Groundwater transport
* Seismic wave propagation

The basic steps required for sensor placement optimization using Chama are shown in :numref:`fig-flowchart`.

.. _fig-flowchart:
.. figure:: figures/flowchart.png
   :scale: 100 %
   :alt: Chama flowchart
   
   Basic steps in sensor placement optimization using Chama.
   
* **Transport simulation**: Generate an ensemble of transport simulations representative of the system in which sensors will be used.
* **Sensor technology**: Define a set of feasible sensor technologies, including stationary and mobile sensors, point detectors and cameras.
* **Impact assessment**: Extract the impact of detecting transport simulations given a set of defined sensor technologies.
* **Optimization**: Optimize sensor location and type given a sensor budget.
* **Interpretation**: Generate maps of the site that include sensor layout and information about scenarios that were and were not detected.

Each step is discussed further in the user manual.
