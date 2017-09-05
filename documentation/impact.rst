.. raw:: latex

    \newpage

.. _impact:
	
Impact assessment
=================

Impact assessment extracts the `impact` if a particular sensor detects a
particular scenario. Impact can be measured using a wide range of metrics.
In Chama, impact assessment starts by extracting the times when each sensor
detects a transport scenario. Detection can be used in a wide range of
sensor placement optimizations, including maximizing coverage or minimizing
detection time. The :mod:`chama.impact` module is used to extract detection
times and convert detection time to other damage metrics.

Chama uses Pandas DataFrames [Mcki13]_ to store the impact assessment. Each
DataFrame has three columns: Scenario, Sensor, and Impact. Exact column names
must be used.  Note that 'Impact' can represent different metrics.

Detection times
---------------
While some scenarios can be detected by a single sensor multiple times, other
scenarios can go undetected by all sensors. In general, detection depends on
the scenario environmental conditions and the sensor location and operating
conditions.

The following example demonstrates how to extract detection times 
using a predefined `signal`, described in the :ref:`transport` section, 
and a set of predefined sensors, described in the :ref:`sensors` section.

Group sensors in a dictionary:

.. doctest::
    :hide:

    >>> from __future__ import print_function
    >>> import chama
    >>> import pandas as pd
    >>> import numpy as np
    >>> posA = chama.sensors.Stationary(location=(1,2,3))
    >>> detA = chama.sensors.Point(sample_times=[0,10,20,30], threshold=30)
    >>> stationary_pt_sensor = chama.sensors.Sensor(position=posA, detector=detA)
    >>> posB = chama.sensors.Mobile(locations=[(2,3,1.1),(2,3,1),(2,2.9,1),(2,3,3)], speed=1)
    >>> detB = chama.sensors.Point(sample_times=[0,10,20,30], threshold=60)
    >>> mobile_pt_sensor = chama.sensors.Sensor(position=posB, detector=detB)
    >>> posC = chama.sensors.Stationary(location=(2,2,3))
	>>> detC = chama.sensors.Camera(threshold=100, sample_times=[0,10,20,30,40], direction=(0,0,-1))
    >>> stationary_camera_sensor = chama.sensors.Sensor(position=posC, detector=detC)
	>>> posD = chama.sensors.Mobile(locations=[(1,1,1),(1,2,1),(1,2,2),(2,1,1)], speed=1)
	>>> detD = chama.sensors.Camera(threshold=10000, sample_times=[0], direction=(1,1,1))
	>>> mobile_camera_sensor = chama.sensors.Sensor(position=posD, detector=detD)
	>>> x,y,z,t = np.meshgrid([1, 2, 3], [1, 2, 3], [1, 2, 3], [0, 10, 20, 30, 40])
    >>> signal = pd.DataFrame({'X': x.flatten(),'Y': y.flatten(),'Z': z.flatten(),'T': t.flatten(),
    ...                        'S1': x.flatten() * t.flatten(),
    ...                        'S2': z.flatten() * t.flatten(),
    ...                        'S3': (t.flatten()-10) * t.flatten()})

.. doctest::

    >>> sensors = {}
    >>> sensors['A'] = stationary_pt_sensor
    >>> sensors['B'] = mobile_pt_sensor
    >>> sensors['C'] = stationary_camera_sensor
    >>> sensors['D'] = mobile_camera_sensor

Extract detection times:

.. doctest::

    >>> det_times = chama.impact.detection_times(signal, sensors)

.. doctest::
    :hide:

    >>> det_times.sort_values(['Scenario','Sensor'], inplace=True)
    >>> det_times.reset_index(inplace=True)
    >>> det_times.drop('index', inplace=True, axis=1)

.. doctest::

    >>> print(det_times)
      Scenario Sensor            Impact
    0       S1      A              [30]
    1       S1      B              [30]
    2       S1      C  [10, 20, 30, 40]
    3       S2      A      [10, 20, 30]
    4       S2      B              [30]
    5       S2      C  [10, 20, 30, 40]
    6       S3      A          [20, 30]
    7       S3      B          [20, 30]
    8       S3      C      [20, 30, 40]

	
The example shows that Scenario S1 was detected by Sensor A at time 30
(units of time depend on the transport simulation). Scenario S1 was also
detected by Sensor B and time 30 and Sensor C at times 20, 30 and 40.
Scenario S2 was detected by Sensors A, B, and C. Scenario S3 was detected by
Sensors A, B, and C. Sensor D did not detect any scenarios.

This information can be used directly to optimization a sensor layout that
maximizes coverage. To optimize a sensor layout that minimizes detection
time, each detected scenario-sensor pair must be represented by a single
detection time.  This can be obtained by taking the minimum, mean, median,
etc. from the list of detection times.

Extract the minimum detection time:

.. doctest::

    >>> min_det_time = chama.impact.detection_time_stats(det_times, 'min')
    >>> print(min_det_time)
      Scenario Sensor Impact
    0       S1      A     30
    1       S1      B     30
    2       S1      C     10
    3       S2      A     10
    4       S2      B     30
    5       S2      C     10
    6       S3      A     20
    7       S3      B     20
    8       S3      C     20


Damage metrics
--------------
Depending on the information available from the transport simulation,
detection time can be converted to other measures of damage, such as damage
cost, extent of contamination, or ability to protect critical assets and
populations. These metrics can be used in sensor placement optimization to
minimize damage. For example, if the cost of detecting scenario S1 at time
20 is $40,000, then the damage metric for that scenario can be translated
from a detection time of 20 to a cost of $40,000. The data associated with
damage is stored in a Pandas DataFrame with one column for time (T) and one
column for each scenario.

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
      Scenario Sensor Impact
    0       S1      A  80000
    1       S1      B  80000
    2       S1      C  10000
    3       S2      A   5000
    4       S2      B  75000
    5       S2      C   5000
    6       S3      A  50000
    7       S3      B  50000
    8       S3      C  50000

	
Note that the `translate` function interpolates based on time, if needed. The
damage metric can be used in sensor placement optimization to minimize damage.
