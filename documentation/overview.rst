.. raw:: latex
	
	\newpage

Overview
========

.. raw:: latex

Continuous or regularly scheduled monitoring has the potential to quickly
identify changes in the environment. However, even with low-cost sensors,
only a limited number of sensors can be used. The physical placement of
these sensors and the sensor technology used can have a large impact on the
performance of a monitoring strategy.

\newline\newline

Chama is an open source python package which includes mixed-integer,
stochastic programming formulations to determine sensor locations and
technology that maximize the effectiveness of the detection program. The
sensor placement optimization methods used in Chama were originally
developed by Sandia National Laboratories and the U.S. Environmental
Protection Agency for water utilities. The basic sensor placement
optimization method included in Chama is based on methods in the Threat
Ensemble Vulnerability Assessment and Sensor Placement Optimization Tool
(TEVA-SPOT) [USEPA12]_ and the Water Security Toolkit (WST) [USEPA15]_.
These tools embed contaminant transport simulations using the water
distribution network model EPANET [Ross00]_with impact assessment and sensor
 placement optimization methods.

Chama was developed to be a general purpose sensor placement optimization
software tool. The software is intended to be used by industry and the
research community. Chama expands on previous methods by allowing the user
to optimize both the location and type of sensors in a monitoring system.
Both stationary and mobile sensors can be defined for detection at a point
or taking into account a field of view from a camera. Optical gas imaging
models from the Fugitive Emissions Abatement Simulation Testbed (FEAST)
[KeRB16]_have been incorporated into Chama. Furthermore, transport
simulations can represent a wide range of applications, including (but not
limited to):

* Air dispersion
* Transport in pipe networks
* Surface water transport
* Seismic wave propagation

The basic steps required for sensor placement optimization using Chama are
shown in :numref:`fig-flowchart`.

.. _fig-flowchart:
.. figure:: figures/flowchart.png
   :scale: 100 %
   :alt: Chama flowchart
   
   Basic steps in sensor placement optimization using Chama.
   
* :ref:`transport`: Generate an ensemble of transport simulations
  representative of the system in which sensors will be used.

* :ref:`sensors`: Define a set of feasible sensor technologies, including
  stationary and mobile sensors, point detectors and cameras.

* :ref:`impact`: Extract the impact of detecting transport simulations given
  a set of defined sensor technologies.

* :ref:`optimization`: Optimize sensor location and type given a sensor
  budget.

* :ref:`graphics`: Generate maps of the site that include sensor layout and
  information about scenarios that were and were not detected.

The following documentation includes additional information on these steps,
installation instructions, description of software features, and software
license.  It is assumed that the reader is familiar with the Python
Programming Language.  References are included for additional background on
software components.
