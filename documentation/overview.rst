.. raw:: latex
	
    \newpage
	\listoffigures
	\newpage
	\pagenumbering{arabic}
	\setcounter{page}{1}

Overview
========

Chama is an open source Python package which includes sensor placement 
optimization methods for a wide range of applications.  
Some of the methods in Chama were originally developed by Sandia 
National Laboratories and the U.S. Environmental Protection Agency to 
design sensor networks to detect contamination in water distribution systems [BHPU06]_ [USEPA12]_ [USEPA15]_. 
In this context, contamination scenarios are 
precomputed using a water distribution system model, feasible sensor locations and thresholds are defined, 
and the optimization method selects a set of sensors to minimize a given objective.

Chama was developed to be a general purpose sensor placement optimization
software tool. 
The software includes mixed-integer
linear programming formulations to determine sensor locations and
technology that maximize monitoring effectiveness. 
The software is intended to be used by regulatory agencies,
industry, and the research community. Chama allows the user to optimize 
both the location and type of sensors
in a monitoring system. Chama includes functionality to define point and
camera sensors that can be stationary or mobile. Furthermore, third party
system models can be integrated into the software to determine sensor placement 
for a wide range of applications.  Example applications are included in :numref:`fig-exapps`.  

.. _fig-exapps:
.. figure:: figures/example_applications.png
   :scale: 100 %
   :align: center
   :alt: Example applications
   
   Example sensor placement applications

For each application, an appropriate model must be selected to represent the system.  For example, 
atmospheric dispersion models can be used to place sensors to monitor oil and gas emissions, while 
water distribution system models can be used to place sensors to monitor drinking water quality.

The basic steps required for sensor placement optimization using Chama are
shown in :numref:`fig-flowchart`.  

.. _fig-flowchart:
.. figure:: figures/flowchart.png
   :scale: 100 %
   :align: center
   :alt: Chama flowchart
   
   Basic steps in sensor placement optimization using Chama
   
* :ref:`simulation`: Generate an ensemble of simulations
  representative of the system in which sensors will be deployed.
* :ref:`sensors`: Define a set of feasible sensor technologies, including
  stationary and mobile sensors, point detectors and cameras.
* :ref:`impact`: Extract the impact of detecting each simulation given
  a set of sensor technologies.
* :ref:`optimization`: Optimize sensor location and type given a sensor
  budget.
* :ref:`graphics`: Generate maps of the site that include the optimal sensor
  layout and information about scenarios that were and were not detected.

The user can enter the workflow at any stage.  For example, if the impact assessment 
was determined using other methods, Chama can still be used to optimize
sensor placement.
The following documentation includes additional information on these steps,
along with installation instructions, software application programming
interface (API), and software license.  It is assumed that the reader is
familiar with the Python Programming Language.  References are included for
additional background on software components.
