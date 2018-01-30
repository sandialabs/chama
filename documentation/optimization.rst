.. raw:: latex

    \newpage

.. _optimization:

Optimization
============

The :mod:`chama.optimize` module contains **Impact**,  **Scenario Coverage**, and **Geographic Coverage** 
sensor placement optimization formulations. 
The formulations are written in Pyomo [HLWW12]_ and solved
using an open source or commercial solvers such as GLPK [Makh10]_,
Gurobi [GUROBI]_, or CPLEX [CPLEX]_.
The open source GLPK solver is used by default. 
Additional optimization formulations could be added to this
module. 

Impact
--------

The Impact formulation is used to determine optimal sensor
placement and type that minimizes impact, where impact can be the sensor's 
detection time or some other measure of damage.
The Impact formulation, which is based on the p-median facility location problem, 
is described below:

.. math::
   
    \text{minimize} \qquad &\sum_{a \in A} \alpha_a \sum_{i \in {\cal L}_a}
    d_{ai} x_{ai}\\
	\text{subject to} \qquad &\sum_{i\in {\cal L}_a} x_{ai} = 1 \hspace{1.2in}
    \forall a \in A\\
	&x_{ai} \le s_i       \hspace{1.47in}  \forall a \in A, i \in {\cal L}_a\\
	&\sum_{i \in L} c_i s_i \le p\\ 
	&s_i \in \{0,1\}      \hspace{1.3in}      \forall i \in L\\ 
	&0 \leq x_{ai} \leq 1 \hspace{1.23in}      \forall a \in A, i \in {\cal L}_a 

where:

* :math:`A` is the set of all scenarios

* :math:`L` is the set of all candidate sensors

* :math:`{\cal L_a}` is the set of all sensors that are capable of detecting
  scenario :math:`a`

* :math:`\alpha_a` is the probability of occurrence for scenario :math:`a`

* :math:`d_{ai}` is the impact coefficient, and represents some measure
  of the impact that will be incurred if scenario :math:`a` is first
  detected by sensor :math:`i`

* :math:`x_{ai}` is an indicator variable that will be 1 if sensor
  :math:`i` is installed and that sensor is the first to detect scenario
  :math:`a` (where `first` is defined as the minimum possible impact,
  usually defined as time to detection)

* :math:`s_i` is a binary variable that will be 1 if sensor :math:`i` is
  selected, and 0 otherwise

* :math:`c_i` is the cost of sensor :math:`i` 

* :math:`p` is the sensors budget

The size of the Impact formulation is determined by the number of
binary variables.  Although :math:`x_{ai}` is a binary indicator
variable, it is relaxed to be continuous between 0 and 1, and yet it
always converges to a value of 0 or 1. Therefore, the number of binary
variables that need to be considered by the solver is a function of the
number of candidate sensors alone, and not the number of scenarios
considered.  This formulation has been used to place sensors in large
water distribution networks [BHPU06]_ [USEPA12]_ [USEPA15]_ and for gas
detection in petrochemical facilities [LBSW12]_.

The user supplies the impact assessment, :math:`d_{ai}`, sensor budget,
:math:`p`, and (optionally) sensor cost, :math:`c_i` and the
scenario probability, :math:`\alpha_a`, as described below:

* Impact assessment: A single value of impact (detection time or other measure of damage) for 
  each sensor that detects a scenario.  Impact is stored as a Pandas DataFrame, 
  as described in the :ref:`impact` section.  
  
* Sensor budget: The number of sensors to place, or total budget for sensors.
  If the 'use_sensor_cost' flag is True, the sensor budget is a dollar amount
  and the optimization uses the cost of individual sensors.  If the
  'use_sensor_cost' flag is False (default), the sensor budget is a number of
  sensors and the optimization does not use sensor cost.

* Sensor characteristics: Sensor characteristics include the cost of each
  sensor. Sensor characteristics are stored as a Pandas DataFrame with columns
  'Sensor' and 'Cost'. Cost is used in the sensor placement optimization if the
  'use_sensor_cost' flag is set to True.
  
* Scenario characteristics: Scenario characteristics include scenario
  probability and the impact for undetected scenarios. Scenario characteristics
  are stored as a Pandas DataFrame with columns 'Scenario', 'Undetected Impact'
  , and 'Probability'. Undetected Impact is required for each scenario. When
  minimizing detection time, the undetected impact value can be set to a value
  larger than time horizon used for the study. Individual scenarios can also be
  given different undetected impact values. Probability is used if the
  'use_scenario_probability' flag is set to True.
  
Results are stored in a dictionary with the following information:

* Sensors: A list of selected sensors

* Objective: The expected (mean) impact based on the selected sensors

* Assessment: The impact value for each sensor-scenario pair.
  The assessment is stored as a Pandas DataFrame with columns 'Scenario', 'Sensor', and 
  'Impact' (same format as the input Impact assessment')
  If the selected sensors did not detect a particular scenario, the impact is set to 
  the Undetected Impact.
  
The following example demonstrates the use of the Impact formulation.

.. doctest::
    :hide:

    >>> import pandas as pd
    >>> import chama
    >>> sensor = pd.DataFrame({'Sensor': ['A', 'B', 'C', 'D'],
    ...                        'Cost': [100.0, 200.0, 500.0, 1500.0]})
    >>> sensor = sensor[['Sensor', 'Cost']]
    >>> scenario = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],
    ...                          'Undetected Impact': [48.0, 250.0, 100.0],
    ...                          'Probability': [0.25, 0.60, 0.15]})
    >>> scenario = scenario[['Scenario', 'Undetected Impact', 'Probability']]
    >>> det_times = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],
    ...                           'Sensor': ['A', 'A', 'B'],
    ...                           'Detection Times': [[2, 3, 4], [3], [4, 5, 6, 7]]})
    >>> det_times = det_times[['Scenario', 'Sensor', 'Detection Times']]
    >>> min_det_time = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],
    ...                              'Sensor': ['A', 'A', 'B'],
    ...                              'Impact': [2.0,3.0,4.0]})
	>>> min_det_time = min_det_time[['Scenario', 'Sensor', 'Impact']]
	
.. doctest::
	
    >>> print(min_det_time)
      Scenario Sensor  Impact
    0       S1      A     2.0
    1       S2      A     3.0
    2       S3      B     4.0
    >>> print(sensor)
      Sensor    Cost
    0      A   100.0
    1      B   200.0
    2      C   500.0
    3      D  1500.0
    >>> print(scenario)
      Scenario  Undetected Impact  Probability
    0       S1               48.0         0.25
    1       S2              250.0         0.60
    2       S3              100.0         0.15
	
    >>> impactsolver = chama.optimize.ImpactSolver()
    >>> results = impactsolver.solve(impact=min_det_time, sensor_budget=200,
    ...                              sensor=sensor, scenario=scenario,
    ...                              use_scenario_probability=True,
    ...                              use_sensor_cost=True)
	
    >>> print(results['Sensors'])
    ['A']
    >>> print(results['Objective'])
    17.3
    >>> print(results['Assessment'])
      Scenario Sensor  Impact
    0       S1      A     2.0
    1       S2      A     3.0
    2       S3   None   100.0


Scenario Coverage
--------------------

The Scenario Coverage formulation is used to place sensors that maximize detection of 
each scenario.
The Scenario Coverage formulation is described below:

.. math::

	\text{maximize} ...

where:

* ...

The user supplies ...

Results are stored in a dictionary with the following information:

* ...

The following example demonstrates the use of the Scenario Coverage formulation.
The results list scenario-time pairs that were detected by the sensor
placement (listed as a (time, scenario) tuple). 

.. doctest::

    >>> print(det_times)
      Scenario Sensor Detection Times
    0       S1      A       [2, 3, 4]
    1       S2      A             [3]
    2       S3      B    [4, 5, 6, 7]
    >>> print(sensor)
      Sensor    Cost
    0      A   100.0
    1      B   200.0
    2      C   500.0
    3      D  1500.0
    >>> print(scenario)
      Scenario  Undetected Impact  Probability
    0       S1               48.0         0.25
    1       S2              250.0         0.60
    2       S3              100.0         0.15
    >>> scenario_time, new_scenario = chama.impact.detection_times_to_coverage(
    ...                                         det_times,
    ...                                         coverage_type='scenario-time',
    ...                                         scenario=scenario)

    >>> print(scenario_time)
      Sensor                          Coverage
    0      A  [S1-2.0, S1-3.0, S1-4.0, S2-3.0]
    1      B  [S3-4.0, S3-5.0, S3-6.0, S3-7.0]
    >>> print(new_scenario)
      Scenario  Undetected Impact  Probability
    0   S1-2.0               48.0         0.25
    1   S1-3.0               48.0         0.25
    2   S1-4.0               48.0         0.25
    3   S2-3.0              250.0         0.60
    4   S3-4.0              100.0         0.15
    5   S3-5.0              100.0         0.15
    6   S3-6.0              100.0         0.15
    7   S3-7.0              100.0         0.15

    >>> coverage = chama.optimize.ScenarioCoverageSolver()
    >>> results = coverage.solve(coverage=scenario_time, sensor_budget=200,
    ...                          sensor=sensor, scenario=new_scenario,
    ...                          use_sensor_cost=True)
	
    >>> print(results['Sensors'])
    ['A']
    >>> print(results['Objective'])
    4.0
    >>> print(results['FractionDetected'])
    0.5
    >>> print(results['SensorAssessment'])  # doctest: +SKIP
    {'A': ['S1-2.0', 'S1-3.0', 'S1-4.0', 'S2-3.0']}
    >>> print(results['EntityAssessment'])  # doctest: +SKIP
    {'S3-6.0': [], 'S3-7.0': [], 'S2-3.0': ['A'], 'S1-4.0': ['A'], 'S3-4.0': [], 'S3-5.0': [], 'S1-3.0': ['A'], 'S1-2.0': ['A']}

..
    The following test checks a subset of the results in the SensorAssessment
    and the EntityAssessment dictionaries. These cannot be tested using the
    above print statements because of Python 2/3 compatibility issues and
    non-deterministic dictionary ordering.
.. doctest::
    :hide:

    >>> print(results['SensorAssessment']['A'])
    ['S1-2.0', 'S1-3.0', 'S1-4.0', 'S2-3.0']
    >>> print(results['EntityAssessment']['S3-6.0'])
    []
    >>> print(results['EntityAssessment']['S3-7.0'])
    []
    >>> print(results['EntityAssessment']['S2-3.0'])
    ['A']
    >>> print(results['EntityAssessment']['S1-4.0'])
    ['A']

Geographic Coverage
---------------------

The Geographic Coverage formulation is used to place sensors that maximize the geographic 
region of detection.
The Geographic Coverage formulation is described below:

.. math::

	\text{maximize} ...

where:

* ...

The user supplies ...

Results are stored in a dictionary with the following information:

* ...

The following example demonstrates the use of the Geographic Coverage formulation.