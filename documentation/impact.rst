.. raw:: latex

    \newpage

.. _impact:
	
Impact assessment
=================

Impact assessment extracts the `impact` of a particular sensor detecting
a particular scenario.  Chama includes two options to compute impact
from a set of transport scenarios and sensor characteristics.  The
impact metrics include: **detection time** and **coverage**.  Depending
on the metric selected, the goal of sensor placement optimization is to
minimize or maximize the impact over all scenarios by selecting a set of
sensors.  In general, a scenario will only be detected by certain
sensors under certain operating conditions.  Likewise, a sensor will not
detect all scenarios.  To pick the best set of sensors for a given set
of scenarios, Chama extracts impact metrics to use in sensor placement
optimization.

Chama uses Pandas DataFrames [Mcki13]_ to store the impact data. The
columns of the impact DataFrame are:

* Scenario
* Sensor
* Impact

The :mod:`chama.impact` module has more information on extracting impact
metrics from a set of transport scenarios and sensor characteristics.

Detection time
--------------
Sensors can be placed to minimize detection time.  Detection time is the
time that a sensor first detects a scenario.  An example `detection time`
impact DataFrame is shown in :numref:`fig-impact-time`.

.. _fig-impact-time:
.. figure:: figures/impacttime.png
   :scale: 50 %
   :alt: impact
   
   Example detection time impact DataFrame.

This impact DataFrame shows that Scenario A was detected by Sensor 1 at
a time of 2.0 (units of time depend on the simulation data).  Scenario A
was also detected by Sensor 2 at time 5.2.  This information is used in
sensor placement optimization to find the set of sensors that minimizes
detection time.  Sensor placement optimization also requires an impact
value for each scenario which represents the impact if that scenario was
undetected.  When minimizing detection time, this undetected impact
value can be set to a value larger than the maximum detection
time. Individual scenarios can also be given different undetected impact
values.

Detection time can be translated into other metrics that might be
available from the set of transport scenarios.  These metrics might
include cost, population impacted, or other measures of damage.  For
example, if the cost of scenario A at time 2.0 is $40,000, then the
impact metric for that detected scenario can be translated from a
detection time of 2.0 to a cost of $40,0000.  Other lines of the impact
DataFrame be translated in a similar manner.

Coverage
--------
Sensors can also be placed to maximize coverage. Coverage is a measure
of how often a sensor detects a scenario (i.e., the probability of
detection).  The coverage metric records scenario-time pairs that are
detected.  An example `coverage` impact DataFrame is shown in
:numref:`fig-impact-cov`.

.. _fig-impact-cov:
.. figure:: figures/impactcov.png
   :scale: 50 %
   :alt: impact
   
   Example coverage impact DataFrame.
 
This impact DataFrame shows that Scenario A at time 2 was detected by
Sensor 1 and Scenario A at time 5.2 was detected by Sensor 2. This is
essentially a reordering of the data extracted for detection time.  This
information is used in sensor placement optimization to find the set of
sensors that maximizes coverage. Because Chama uses a minimization
function in the optimization routine, the impact metric is 0 if the
scenario-time pair is detected. Sensor placement optimization also
requires an impact value for each scenario-time that went undetected.
When maximizing coverage, this value is set to 1 for each scenario-time
pair.

The following example demonstrates how to extract detection time 
using a `signal`, described in the :ref:`transport` section, 
and a set of sensors, described in the :ref:`sensors` section.
Sensors must be grouped in a dictionary, each with a unique name.  
The dictionary of sensors can be created as follows:

.. doctest::
    :hide:

    >>> import chama
    >>> import pandas as pd
    >>> import numpy as np
	>>> stationary_pt_sensor = chama.sensors.Sensor(sample_times=[0], location=(1,1,1),threshold=0)
	>>> mobile_pt_sensor = chama.sensors.Sensor(sample_times=[0], location=(1,1,1),threshold=0)
	>>> stationary_camera_sensor = chama.sensors.Sensor(sample_times=[0], location=(1,1,1),threshold=0)
	>>> mobile_camera_sensor = chama.sensors.Sensor(sample_times=[0], location=(1,1,1),threshold=0)
	>>> x,y,z,t = np.meshgrid([1,2], [1,2], [1,2], [0,10])       
    >>> signal = pd.DataFrame({'X': x.flatten(),'Y': y.flatten(), 'Z': z.flatten(),'T': t.flatten(),'S': x.flatten()})
	
.. doctest::

    >>> sensors = {}
    >>> sensors['sensor1'] = stationary_pt_sensor
    >>> sensors['sensor2'] = mobile_pt_sensor
    >>> sensors['sensor3'] = stationary_camera_sensor
    >>> sensors['sensor4'] = mobile_camera_sensor
	
Detection time can then be extracted using the following code:

.. doctest::

    >>> impact = chama.impact.extract(signal, sensors)
