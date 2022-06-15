.. raw:: latex

    \newpage

.. _impact:
	
Impact assessment
=================
Impact assessment extracts the **impact** of a particular sensor detecting a
particular scenario. Impact can be measured using a variety of metrics such as
time to detection, population impacted, or volume of contaminant released
before detection. Additionally, these impact metrics can be used to define
when a sensor covers a particular scenario for use in
coverage-based optimization formulations.

The :mod:`chama.impact` module converts information about the signal and
sensors, described in the :ref:`simulation` and :ref:`sensors` sections, into
the input needed for the sensor placement optimization formulations described in the
:ref:`optimization` section.
Impact assessment starts by extracting the times when each sensor
detects a scenario. After that, detection times can be converted into other impact
metrics used in the :ref:`impactform` or
coverage-based formats used in the :ref:`coverageform`.

Extract detection times
-----------------------
The ability for a sensor to detect a scenario depends on several factors, including the scenario environmental conditions,
sensor location, and sensor operating parameters. While some scenarios might
be detected multiple times by a single sensor, other scenarios can go
undetected by all sensors.
The following example demonstrates how to extract detection times using 
a predefined signal, and a set of predefined sensors.

Obtain a signal DataFrame and group sensors (defined in the :ref:`sensors`
section) in a dictionary:

.. doctest::
    :hide:

    >>> from __future__ import print_function, division
    >>> import chama
    >>> import pandas as pd
    >>> import numpy as np
    >>> pd.set_option('display.max_columns', 20)
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
    >>> signal = signal[['S1', 'S2', 'S3', 'T', 'X', 'Y', 'Z']]
	
.. doctest::

    >>> print(signal.head())
       S1  S2    S3   T  X  Y  Z
    0   0   0     0   0  1  1  1
    1  10  10     0  10  1  1  1
    2  20  20   200  20  1  1  1
    3  30  30   600  30  1  1  1
    4  40  40  1200  40  1  1  1

    >>> sensors = dict()
    >>> sensors['A'] = stationary_pt_sensor
    >>> sensors['B'] = mobile_pt_sensor
    >>> sensors['C'] = stationary_camera_sensor
    >>> sensors['D'] = mobile_camera_sensor

Extract detection times:

.. doctest::

    >>> det_times = chama.impact.extract_detection_times(signal, sensors)

.. doctest::
    :hide:

    >>> det_times.sort_values(['Scenario','Sensor'], inplace=True)
    >>> det_times.reset_index(inplace=True)
    >>> det_times.drop('index', inplace=True, axis=1)

.. doctest::

    >>> print(det_times)
      Scenario Sensor   Detection Times
    0       S1      A              [30]
    1       S1      B              [30]
    2       S1      C  [10, 20, 30, 40]
    3       S2      A      [10, 20, 30]
    4       S2      B          [20, 30]
    5       S2      C  [10, 20, 30, 40]
    6       S3      A          [20, 30]
    7       S3      B          [20, 30]
    8       S3      C      [20, 30, 40]

	
The example shows that Scenario S1 was detected by Sensor A at time 30
(units of time depend on the simulation). Scenario S1 was also detected by
Sensor B and time 30 and Sensor C at times 10, 20, 30 and 40. Scenario S2 was
detected by Sensors A, B, and C. Scenario S3 was detected by Sensors A, B, and
C. Sensor D did not detect any scenarios.

The detection times DataFrame can be converted into the required input
format for the :ref:`impactform` or :ref:`coverageform` as described below.

Convert detection times to input for the Impact Formulation
-----------------------------------------------------------
The :ref:`impactform` requires as input
a DataFrame with three columns: 'Scenario', 'Sensor', and 'Impact', where
the 'Impact' is a single numerical value for each row. This means that the
list of detection times in the DataFrame produced above must be reduced to a
single numerical value representing the impact to be minimized.

Minimum detection time
......................

The example below shows how to build an input DataFrame for the :ref:`impactform` to
optimize a sensor layout that minimizes detection time.

Extract detection time statistics:

.. doctest::

    >>> det_time_stats = chama.impact.detection_time_stats(det_times)
    >>> print(det_time_stats)
      Scenario Sensor  Min  Mean  Median  Max  Count
    0       S1      A   30  30.0    30.0   30      1
    1       S1      B   30  30.0    30.0   30      1
    2       S1      C   10  25.0    25.0   40      4
    3       S2      A   10  20.0    20.0   30      3
    4       S2      B   20  25.0    25.0   30      2
    5       S2      C   10  25.0    25.0   40      4
    6       S3      A   20  25.0    25.0   30      2
    7       S3      B   20  25.0    25.0   30      2
    8       S3      C   20  30.0    30.0   40      3

Extract the minimum detection time from the statistics computed above:

.. doctest::

    >>> min_det_time = det_time_stats[['Scenario','Sensor','Min']]
    >>> min_det_time = min_det_time.rename(columns={'Min':'Impact'})
    >>> print(min_det_time)
      Scenario Sensor  Impact
    0       S1      A      30
    1       S1      B      30
    2       S1      C      10
    3       S2      A      10
    4       S2      B      20
    5       S2      C      10
    6       S3      A      20
    7       S3      B      20
    8       S3      C      20

Other impact metrics
....................
Depending on the information available from the simulation, detection time
can be converted to other measures of impact, such as damage cost, extent of
contamination, or ability to protect critical assets and populations. For
example, if the cost of detecting scenario S1 at time 30 is $80,000, then the
impact metric for that scenario can be translated from a detection time of 30
to a cost of $80,000. The data associated with the new impact metric is stored in a Pandas
DataFrame with one column for time, 'T', and one column for each scenario (name
specified by the user).

Example impact costs associated with each scenario and time:

.. doctest::
    :hide:

    >>> impact_cost = pd.DataFrame({'T': [0, 10, 20, 30, 40],'S1': [0, 10000, 40000, 80000, 100000],'S2': [0, 5000, 20000, 75000, 90000],'S3': [0, 15000, 50000, 95000, 150000]})
    >>> impact_cost = impact_cost[['T', 'S1','S2', 'S3']]
    

.. doctest::

    >>> print(impact_cost)
        T      S1     S2      S3
    0   0       0      0       0
    1  10   10000   5000   15000
    2  20   40000  20000   50000
    3  30   80000  75000   95000
    4  40  100000  90000  150000

Rename the time column in min_det_time to 'T':

.. doctest::

    >>> det_time = min_det_time.rename(columns={'Impact':'T'}, inplace=False)
    >>> print(det_time)
      Scenario Sensor   T
    0       S1      A  30
    1       S1      B  30
    2       S1      C  10
    3       S2      A  10
    4       S2      B  20
    5       S2      C  10
    6       S3      A  20
    7       S3      B  20
    8       S3      C  20
	

Convert minimum detection time to damage cost:

.. doctest::

    >>> impact_metric = chama.impact.detection_time_to_impact(det_time, impact_cost)
    >>> print(impact_metric)
      Scenario Sensor  Impact
    0       S1      A   80000
    1       S1      B   80000
    2       S1      C   10000
    3       S2      A    5000
    4       S2      B   20000
    5       S2      C    5000
    6       S3      A   50000
    7       S3      B   50000
    8       S3      C   50000

Note that the
:py:meth:`detection_time_to_impact<chama.impact.detection_time_to_impact>`
function interpolates based on time, if needed. 

Convert detection times to input for the Coverage Formulation
-------------------------------------------------------------
The :ref:`coverageform` requires as input
a DataFrame with two columns: 'Sensor', and 'Coverage', where the 'Coverage' is
a list of entities covered by each sensor. The formulation optimizes a sensor
layout that maximizes the coverage of the entities contained in this
DataFrame.
An `entity` to be covered might include scenarios, scenario-time pairs, or
geographic locations. 

Scenario coverage
.................
The following example converts detection times to scenario coverage. 
With `scenario` coverage, a scenario is the entity to be covered. A scenario
is considered covered by a sensor if that sensor detects that scenario at
any time.

Recall the detection times DataFrame from above:

.. doctest::

    >>> print(det_times)
      Scenario Sensor   Detection Times
    0       S1      A              [30]
    1       S1      B              [30]
    2       S1      C  [10, 20, 30, 40]
    3       S2      A      [10, 20, 30]
    4       S2      B          [20, 30]
    5       S2      C  [10, 20, 30, 40]
    6       S3      A          [20, 30]
    7       S3      B          [20, 30]
    8       S3      C      [20, 30, 40]

Convert detection times to `scenario` coverage:

.. doctest::

    >>> scenario_cov = chama.impact.detection_times_to_coverage(det_times, coverage_type='scenario')
    >>> print(scenario_cov)
      Sensor      Coverage
    0      A  [S1, S2, S3]
    1      B  [S1, S2, S3]
    2      C  [S1, S2, S3]

This example shows that sensor A covers the scenarios S1, S2, and S3.
Sensors B and C also cover all three scenarios.

Scenario-time coverage
......................

The next example converts detection times to scenario-time coverage. 
With `scenario-time` coverage, the entities to be covered are all combinations
of the scenarios and the detection times. This type of coverage gives more
weight to sensors that detect scenarios for longer periods of time.
The same
:py:meth:`detection_times_to_coverage<chama.impact.detection_times_to_coverage>`
function can be used to convert detection times to scenario-time coverage
with one major difference to the previous case. With `scenario` coverage the
scenarios themselves become the entities to be covered. This means that if
there is additional data available for the scenarios such as
weights/probabilities or undetected impact, these values can be used
directly in the coverage solver. With `scenario-time` coverage, we are
essentially defining new entities/scenarios. So any data corresponding to the
original scenarios must be translated to the new entities before they can be
passed to the coverage solver. The
:py:meth:`detection_times_to_coverage<chama.impact.detection_times_to_coverage>`
function does this by accepting an optional 'scenario' keyword argument
containing a DataFrame with scenario probabilities and undetected impact. These
values are then propagated to the new scenario-time entities and a new
DataFrame is returned with this information. 

Convert detection times to `scenario-time` coverage and propagate scenario
information to new scenario-time pairs:

.. doctest::
    :hide:

    >>> scenario = pd.DataFrame({'Scenario': ['S1','S2','S3'], 'Probability': [0.25,0.5,0.75], 'Undetected Impact': [100,100,100]})
    >>> scenario = scenario[['Probability', 'Scenario', 'Undetected Impact']]
	
.. doctest::

    >>> print(scenario)
       Probability Scenario  Undetected Impact
    0         0.25       S1                100
    1         0.50       S2                100
    2         0.75       S3                100

    >>> scen_time_cov, new_scenario = chama.impact.detection_times_to_coverage(
    ...                                          det_times,
    ...                                          coverage_type='scenario-time',
    ...                                          scenario=scenario)
	
    >>> print(scen_time_cov)
      Sensor                                           Coverage
    0      A  [S1-30.0, S2-10.0, S2-20.0, S2-30.0, S3-20.0, ...
    1      B      [S1-30.0, S2-20.0, S2-30.0, S3-20.0, S3-30.0]
    2      C  [S1-10.0, S1-20.0, S1-30.0, S1-40.0, S2-10.0, ...

    >>> print(new_scenario)
       Scenario  Probability  Undetected Impact
    0   S1-30.0         0.25                100
    2   S1-10.0         0.25                100
    3   S1-20.0         0.25                100
    5   S1-40.0         0.25                100
    6   S2-10.0         0.50                100
    7   S2-20.0         0.50                100
    8   S2-30.0         0.50                100
    14  S2-40.0         0.50                100
    15  S3-20.0         0.75                100
    16  S3-30.0         0.75                100
    21  S3-40.0         0.75                100


This example shows that sensor A covers the scenario-time pairs S1-30.0,
S2-10.0, and S2-20.0 among others. In addition, notice that the probability
and undetected impact for scenario S1 is propagated to all scenario-time
pairs containing S1 in the new_scenario DataFrame.

Convert input for the Impact Formulation to the Coverage Formulation
--------------------------------------------------------------------
Users can also convert the input DataFrame for the :ref:`impactform`
to the input DataFrame for the :ref:`coverageform`. This is
especially convenient in cases where the user is solving optimization
problems using both solver classes and the DataFrame for the impact
solver was generated outside of the standard Chama workflow (i.e. the
signal, sensors, or detection_times DataFrames are unavailable).
In the following example, an impact DataFrame is converted to 
a `scenario` coverage DataFrame.

Recall the impact DataFrame containing minimum detection time from above:

.. doctest::

    >>> print(min_det_time)
      Scenario Sensor  Impact
    0       S1      A      30
    1       S1      B      30
    2       S1      C      10
    3       S2      A      10
    4       S2      B      20
    5       S2      C      10
    6       S3      A      20
    7       S3      B      20
    8       S3      C      20
	
Convert the impact DataFrame to a coverage DataFrame:

.. doctest::

    >>> scenario_cov = chama.impact.impact_to_coverage(min_det_time)
    >>> print(scenario_cov)
      Sensor      Coverage
    0      A  [S1, S2, S3]
    1      B  [S1, S2, S3]
    2      C  [S1, S2, S3]

Notice that we end up with the same scenario coverage DataFrame as before
but using different input.