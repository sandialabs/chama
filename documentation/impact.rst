.. raw:: latex

    \newpage

.. _impact:
	
Impact assessment
=================

Impact assessment extracts the `impact` if a particular sensor detects a particular scenario. 
Impact can be measured using a wide range of metrics. 
In Chama, impact assessment starts by extracting the times when each sensor detects a 
transport scenario. Detection can be used in a wide range of sensor placement optimizations, 
including maximizing coverage or minimizing detection time.  
The :mod:`chama.impact` module is used to extract detection 
times and convert detection time to other damage metrics.

Chama uses Pandas DataFrames [Mcki13]_ to store the impact assessment.  
Each DataFrame has three columns: Scenario, Sensor, and Impact value.
Exact column names must be used for Scenario and Sensor. 
The column name for impact value can be defined by the user.

Detection times
-----------------
While some scenarios can be detected by a single sensor multiple times, 
other scenarios can go undetected by all sensors. In general, detection depends on the 
scenario environmental conditions and the sensor location and operating conditions. 

The following example demonstrates how to extract detection times 
using a predefined `signal`, described in the :ref:`transport` section, 
and a set of predefined sensors, described in the :ref:`sensors` section.

Group sensors in a dictionary:

.. doctest::
    :hide:

    >>> import chama
    >>> import pandas as pd
    >>> import numpy as np
    >>> stationary_pt_sensor = chama.sensors.Sensor(sample_times=[0,10,20],location=(1, 2, 3),threshold=15)
    >>> mobile_pt_sensor = chama.sensors.Sensor(sample_times=[0,10,20],location=(2, 3, 1),threshold=25)
    >>> stationary_camera_sensor = chama.sensors.Sensor(sample_times=[0,10,20,30,40],location=(3, 2, 1),threshold=100)
    >>> mobile_camera_sensor = chama.sensors.Sensor(sample_times=[0],location=(1, 2, 1),threshold=1000)     
    >>> x, y, z, t = np.meshgrid([1, 2, 3], [1, 2, 3], [1, 2, 3], [0, 10, 20, 30, 40])
    >>> signal = pd.DataFrame({'X': x.flatten(),'Y': y.flatten(),'Z': z.flatten(),'T': t.flatten(),'S1': x.flatten() * t.flatten(),'S2': z.flatten() * t.flatten(),'S3': t.flatten() * t.flatten()})

.. doctest::

    >>> sensors = {}
    >>> sensors['A'] = stationary_pt_sensor
    >>> sensors['B'] = mobile_pt_sensor
    >>> sensors['C'] = stationary_camera_sensor
    >>> sensors['D'] = mobile_camera_sensor
	
Extract detection times:

.. doctest::

    >>> det_times = chama.impact.detection_times(signal, sensors)
    >>> print(det_times)
      Scenario Sensor                 T
    0       S1      A              [20]
    1       S1      B              [20]
    2       S1      C              [40]
    3       S2      A          [10, 20]
    4       S3      A          [10, 20]
    5       S3      B          [10, 20]
    6       S3      C  [10, 20, 30, 40]
	
The example shows that Scenario S1 was detected by Sensor A at
time 20 (units of time depend on the transport simulation).  
Scenario S1 was also detected by Sensors B and C.
Scenario S2 was only detected by Sensor A, at times 10 and 20.
Scenario S3 was detected by Sensors A, B, and C, at multiple times.  
Sensor D did not detect any scenarios.

This information can be used directly to optimization a sensor layout that maximizes coverage.
To optimize a sensor layout that minimizes detection time, each detected scenario-sensor pair must be 
represented by a single detection time.  This can be obtained by taking the minimum, mean, median, etc.
from the list of detection times.

Extract the minimum detection time:

.. doctest::

    >>> min_det_time = chama.impact.detection_time_stats(det_times, 'min')
    >>> print(min_det_time)
      Scenario Sensor minT
    0       S1      A   20
    1       S1      B   20
    2       S1      C   40
    3       S2      A   10
    4       S3      A   10
    5       S3      B   10
    6       S3      C   10
	
Damage metrics
-----------------
Depending on the information available from the transport simulation, 
detection time can be converted to other measures of damage, such as 
damage cost, extent of contamination, or ability to protect critical assets and populations.  
These metrics can be used in sensor placement optimization to minimize damage.
For example, if the cost of detecting scenario S1 at time 20 is $40,000, then the
damage metric for that scenario can be translated from a
detection time of 20 to a cost of $40,000. 
The data associated with damage is stored in a Pandas DataFrame with one column for time (T) and 
one column for each scenario.

Example damage costs, associated with each scenario and time:

.. doctest::
    :hide:

    >>> damage_cost = pd.DataFrame({'T': [0, 10, 20, 30, 40],'S1': [0, 10000, 40000, 80000, 100000],'S2': [0, 5000, 20000, 75000, 90000],'S3': [0, 15000, 50000, 95000, 150000]})
    >>> damage_cost = damage_cost[['T', 'S1','S2', 'S3']]
	
.. doctest::

    >>> print(damage_cost)
        T      S1     S2      S3
    0   0       0      0       0
    1  10   10000   5000   15000
    2  20   40000  20000   50000
    3  30   80000  75000   95000
    4  40  100000  90000  150000
	
Convert detection time to damage cost:

.. doctest::

    >>> damage_metric = chama.impact.translate(min_det_time, damage_cost)
    >>> print(damage_metric)
      Scenario Sensor  Damage
    0       S1      A   40000
    1       S1      B   40000
    2       S1      C  100000
    3       S2      A    5000
    4       S3      A   15000
    5       S3      B   15000
    6       S3      C   15000
	
Note that the 'translate' function interpolates based on time (T), if needed.
The damage metric can be used in sensor placement optimization to minimize damage.

