.. raw:: latex

    \newpage

.. _optimization:

Optimization
============

The :mod:`chama.optimize` module contains **Impact** and **Coverage** sensor
placement optimization formulations. The formulations are written in Pyomo
[HLWW12]_ and solved using an open source or commercial solver such as GLPK
[Makh10]_, Gurobi [GUROBI]_, or CPLEX [CPLEX]_. The open source GLPK solver is
used by default. Additional optimization formulations could be added to this
module. 

.. _impactform:

Impact Formulation
------------------

The Impact formulation is used to determine optimal sensor placement and
type that minimizes impact, where impact can be the sensor's detection time
or some other measure of damage. The Impact formulation, which is based on
the p-median facility location problem, is described below:

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

* :math:`d_{ai}` is the impact assessment, and represents some measure
  of the impact that will be incurred if scenario :math:`a` is first
  detected by sensor :math:`i`

* :math:`x_{ai}` is an indicator variable that will be 1 if sensor
  :math:`i` is installed and that sensor is the first to detect scenario
  :math:`a` (where `first` is defined as the minimum possible impact,
  usually defined as time to detection)

* :math:`s_i` is a binary variable that will be 1 if sensor :math:`i` is
  selected, and 0 otherwise

* :math:`c_i` is the cost of sensor :math:`i`

* :math:`p` is the sensor budget

The size of the Impact formulation is determined by the number of binary
variables. Although :math:`x_{ai}` is a binary indicator variable, it is
relaxed to be continuous between 0 and 1, and yet it always converges to a
value of 0 or 1. Therefore, the number of binary variables that need to be
considered by the solver is a function of the number of candidate sensors
alone, and not the number of scenarios considered.  This formulation has been
used to place sensors in large water distribution networks [BHPU06]_ [USEPA12]_
[USEPA15]_ and for gas detection in petrochemical facilities [LBSW12]_.

To use this formulation in Chama, create an
:py:class:`ImpactFormulation<chama.optimize.ImpactFormulation>` object and
specify the impact assessment, :math:`d_{ai}`, sensor budget, :math:`p`, and
(optionally) sensor cost, :math:`c_i` and the scenario probability,
:math:`\alpha_a`, as described below:

* Impact assessment: A single value of impact (detection time or other measure
  of damage) for each sensor that detects a scenario.  Impact is stored as a
  Pandas DataFrame, as described in the :ref:`impact` section.

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

* FractionDetected: The fraction of scenarios that were detected

* TotalSensorCost: Total cost of the selected sensors

* Assessment: The impact value for each sensor-scenario pair.
  The assessment is stored as a Pandas DataFrame with columns 'Scenario',
  'Sensor', and 'Impact' (same format as the input Impact assessment')
  If the selected sensors did not detect a particular scenario, the impact is
  set to the Undetected Impact.
			  
The following example demonstrates the use of the Impact Formulation.

.. doctest::
    :hide:

    >>> import pandas as pd
    >>> import chama
    >>> sensor = pd.DataFrame({'Sensor': ['A', 'B', 'C', 'D'],
    ...                        'Cost': [100.0, 200.0, 400.0, 500.0]})
    >>> sensor = sensor[['Sensor', 'Cost']]
    >>> scenario = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3', 'S4', 'S5'],
    ...                          'Undetected Impact': [50.0, 250.0, 100.0, 75.0, 225.0],
    ...                          'Probability': [0.15, 0.50, 0.05, 0.20, 0.10]})
    >>> scenario = scenario[['Scenario', 'Undetected Impact', 'Probability']]
    >>> det_times = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3', 'S4', 'S5', 'S5'],
    ...                           'Sensor': ['A', 'A', 'B', 'C', 'B', 'D'],
    ...                           'Detection Times': [[2, 3, 4], [3], [4, 5, 6, 7], [1, 3], [6], [2, 4, 6]]})
    >>> det_times = det_times[['Scenario', 'Sensor', 'Detection Times']]
    >>> min_det_time = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3', 'S4', 'S5'],
    ...                              'Sensor': ['A', 'A', 'B', 'C', 'D'],
    ...                              'Impact': [2.0,3.0,4.0,1.0,2.0]})
    >>> min_det_time = min_det_time[['Scenario', 'Sensor', 'Impact']]
	
.. doctest::
	
    >>> print(min_det_time)
      Scenario Sensor  Impact
    0       S1      A     2.0
    1       S2      A     3.0
    2       S3      B     4.0
    3       S4      C     1.0
    4       S5      D     2.0
    >>> print(sensor)
      Sensor   Cost
    0      A  100.0
    1      B  200.0
    2      C  400.0
    3      D  500.0
    >>> print(scenario)
      Scenario  Undetected Impact  Probability
    0       S1               50.0         0.15
    1       S2              250.0         0.50
    2       S3              100.0         0.05
    3       S4               75.0         0.20
    4       S5              225.0         0.10
	
    >>> impactform = chama.optimize.ImpactFormulation()
    >>> results = impactform.solve(impact=min_det_time, sensor_budget=1000,
    ...                              sensor=sensor, scenario=scenario,
    ...                              use_scenario_probability=True,
    ...                              use_sensor_cost=True)
	
    >>> print(results['Sensors'])
    ['A', 'C', 'D']
    >>> print(results['Objective'])
    7.2
    >>> print(results['Assessment'])
      Scenario Sensor  Impact
    0       S1      A     2.0
    1       S2      A     3.0
    2       S4      C     1.0
    3       S5      D     2.0
    4       S3   None   100.0


.. _coverageform:
	
Coverage Formulation
--------------------

The Coverage formulation is used to place sensors that maximize the
coverage of a set of entities, where an entity can be a scenario, scenario-time
pair, or geographic location. The Coverage formulation is described below:

.. math::

    \text{maximize} \qquad &\sum_{a \in A} \alpha_a x_a \\
    \text{subject to} \qquad &x_{a} \le \sum_{i \in {\cal L}_a} s_i
    \hspace{1.15in} \forall a \in A\\
	&\sum_{i \in L} c_i s_i \le p\\
	&s_i \in \{0,1\}      \hspace{1.3in}    \forall i \in L\\
	&0 \leq x_{a} \leq 1 \hspace{1.25in}    \forall a \in A

where:

* :math:`A` is the set of all entities

* :math:`L` is the set of all candidate sensors

* :math:`{\cal L_a}` is the set of all sensors that cover entity :math:`a`

* :math:`\alpha_a` is the objective weight of entity :math:`a`

* :math:`x_{a}` is an indicator variable that will be 1 if entity :math:`a`
  is covered

* :math:`s_i` is a binary variable that will be 1 if sensor :math:`i` is
  selected, and 0 otherwise

* :math:`c_i` is the cost of sensor :math:`i`

* :math:`p` is the sensor budget

This formulation is similar to the Impact formulation in that the number of
binary variables is a function of the number of candidate sensors and not the
number of entities considered.

To use this formulation in Chama, create a
:py:class:`CoverageFormulation<chama.optimize.CoverageFormulation>` object and
specify the coverage, :math:`{\cal L_a}`, sensor budget, :math:`p`, and
(optionally) sensor cost, :math:`c_i` and the entity weights,
:math:`\alpha_a`, as described below:

* Coverage: A list of entities that are covered by a single sensor. Coverage
  is stored as a Pandas DataFrame, as described in the :ref:`impact` section.

* Sensor budget: The number of sensors to place, or total budget for sensors.
  If the 'use_sensor_cost' flag is True, the sensor budget is a dollar amount
  and the optimization uses the cost of individual sensors.  If the
  'use_sensor_cost' flag is False (default), the sensor budget is a number of
  sensors and the optimization does not use sensor cost.

* Sensor characteristics: Sensor characteristics include the cost of each
  sensor. Sensor characteristics are stored as a Pandas DataFrame with columns
  'Sensor' and 'Cost'. Cost is used in the sensor placement optimization if the
  'use_sensor_cost' flag is set to True.

* Entity characteristics: Entity weights stored as a Pandas DataFrame with
  columns 'Entity' and 'Weight'. Weight is used if the 'use_entity_weight' flag
  is set to True.

Results are stored in a dictionary with the following information:

* Sensors: A list of selected sensors

* Objective: The mean coverage based on the selected sensors

* FractionDetected: The fraction of entities that are detected

* TotalSensorCost: Total cost of selected sensors

* EntityAssessment: A dictionary whose keys are the entity names and values
  are a list of sensors that detect that entity

* SensorAssessment: A dictionary whose keys are the sensor names and values
  are the list of entities that are detected by that sensor

The following example demonstrates the use of the Coverage Formulation to solve for
scenario-time coverage. The results list scenario-time pairs that were detected
by the sensor placement (listed as 'scenario-time').

.. doctest::

    >>> print(det_times)
      Scenario Sensor Detection Times
    0       S1      A       [2, 3, 4]
    1       S2      A             [3]
    2       S3      B    [4, 5, 6, 7]
    3       S4      C          [1, 3]
    4       S5      B             [6]
    5       S5      D       [2, 4, 6]
    >>> print(sensor)
      Sensor   Cost
    0      A  100.0
    1      B  200.0
    2      C  400.0
    3      D  500.0
    >>> print(scenario)
      Scenario  Undetected Impact  Probability
    0       S1               50.0         0.15
    1       S2              250.0         0.50
    2       S3              100.0         0.05
    3       S4               75.0         0.20
    4       S5              225.0         0.10
    >>> scenario_time, new_scenario = chama.impact.detection_times_to_coverage(
    ...                                         det_times,
    ...                                         coverage_type='scenario-time',
    ...                                         scenario=scenario)

    >>> print(scenario_time)
      Sensor                                  Coverage
    0      A          [S1-2.0, S1-3.0, S1-4.0, S2-3.0]
    1      B  [S3-4.0, S3-5.0, S3-6.0, S3-7.0, S5-6.0]
    2      C                          [S4-1.0, S4-3.0]
    3      D                  [S5-2.0, S5-4.0, S5-6.0]
    >>> print(new_scenario)
       Scenario  Undetected Impact  Probability
    0    S1-2.0               50.0         0.15
    1    S1-3.0               50.0         0.15
    2    S1-4.0               50.0         0.15
    3    S2-3.0              250.0         0.50
    4    S3-4.0              100.0         0.05
    5    S3-5.0              100.0         0.05
    6    S3-6.0              100.0         0.05
    7    S3-7.0              100.0         0.05
    8    S4-1.0               75.0         0.20
    9    S4-3.0               75.0         0.20
    10   S5-6.0              225.0         0.10
    11   S5-2.0              225.0         0.10
    12   S5-4.0              225.0         0.10

    >>> new_scenario = new_scenario.rename(columns={'Scenario':'Entity',
    ...                                             'Probability':'Weight'})
    >>> coverageform = chama.optimize.CoverageFormulation()
    >>> results = coverageform.solve(coverage=scenario_time, sensor_budget=1000,
    ...                          sensor=sensor, entity=new_scenario,
    ...                          use_sensor_cost=True)
	
    >>> print(results['Sensors'])
    ['A', 'B', 'C']
    >>> print(results['Objective'])
    11.0
    >>> print(round(results['FractionDetected'],2))
    0.85

Grouping Constraints
----------------------------

Constraints can be added to both the Impact and Coverage formulations to enforce or 
restrict the number of sensors allowed from certain sets. These grouping 
constraints take the following general form:

.. math::

    g_{min} \le \sum_{i \in L_g} s_i \le g_{max}

where:

* :math:`L_g` is a subset of all candidate sensors
* :math:`s_i` is a binary variable that will be 1 if sensor :math:`i` is selected, and 0 otherwise	
* :math:`g_{min}` is the minimum number of sensors that must be selected from the subset :math:`L_g`
* :math:`g_{max}` is the maximum number of sensors that may be selected from the subset :math:`L_g`

Grouping constraints can be used to ensure that an optimal sensor placement follows required 
policies or meets practical limitations. For example, you might want to determine the optimal 
sensor placement, while also ensuring that there is at least one sensor in every 10 m x 10 m 
subvolume of the space. This can be formulated by defining sensor subsets :math:`L_g` containing the 
candidate sensors within each subvolume and adding a grouping constraint over each of these 
subsets with :math:`g_{min}` set to 1. 

Another example where grouping constraints might be used is when you have different categories 
of sensors and you want to make sure that an optimal placement has a certain number of each 
category. In this case, you would define a sensor subset :math:`L_g` for each category of sensor and 
then set :math:`g_{min}` and :math:`g_{max}` according to how many sensors you want in each category. 

While grouping constraints are very useful, it should be noted that it is possible to formulate 
infeasible optimization problems if these constraints are not used carefully. 

The following example adds grouping constraints to the Impact formulation.  
This requires the user to 
1) create the Pyomo model, 
2) add the grouping constraints, 
3) solve the model, and 
4) extract the solution summary.

.. doctest::
	
    >>> impactform = chama.optimize.ImpactFormulation()
    
    >>> model = impactform.create_pyomo_model(impact=min_det_time, sensor=sensor, scenario=scenario)
    >>> impactform.add_grouping_constraint(['A', 'B'], min_select=1)
    >>> impactform.add_grouping_constraint(['C', 'D'], min_select=1)
    >>> impactform.solve_pyomo_model(sensor_budget=2)
    >>> results = impactform.create_solution_summary()
    
    >>> print(results['Sensors'])
    ['A', 'D']

Grouping constraints can be added to the Coverage formulation in a similar manner.

