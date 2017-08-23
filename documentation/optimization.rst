.. raw:: latex

    \newpage

.. _optimization:

Optimization
===========================

The :mod:`chama.optimize` module contains **coverage** and **P-median** sensor 
placement optimization formulations.  Additional methods could be added to this module.
Data requirements for sensor placement are listed below, note that some formulations 
require slightly different input:

* Impact assessment: as described in the :ref:`impact` section.

  * For coverage, 
  * For P-median, 

* Sensor budget: Number of sensors to place, or dollar amount for sensors.

* Sensor characteristics

  * Cost
  
* Scenario characteristics
 
  * Probability
  * Undetected impact: Sensor placement optimization also requires an impact
    value for each scenario which represents the impact if that scenario was
    undetected.  When minimizing detection time, this undetected impact
    value can be set to a value larger than the maximum detection
    time. Individual scenarios can also be given different undetected impact
    values.

Coverage
-----------
The following coverage formulation is used to determine the optimal sensor
placement and type that maximizes scenario or time coverage:

.. math::

	equation
	
where:

* 

This does NOT require Undetected Impact?  It is always == 1?

The following example...

.. doctest::
    :hide:

    >>> import pandas as pd
    >>> import chama
    >>> sensor = pd.DataFrame({'Sensor': ['A', 'B', 'C', 'D'],'Cost': [100.0, 200.0, 500.0, 1500.0]})
    >>> sensor = sensor[['Sensor', 'Cost']]
    >>> scenario = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],'Undetected Impact': [48.0, 250.0, 100.0], 'Probability': [0.25, 0.60, 0.15]})
	>>> scenario = scenario[['Scenario', 'Undetected Impact', 'Probability']]
    >>> det_times = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],'Sensor': ['A', 'A', 'B'], 'Impact': [[2,3,4],[3],[4,5,6,7]]})
	>>> min_det_time = pd.DataFrame({'Scenario': ['S1', 'S2', 'S3'],'Sensor': ['A', 'A', 'B'], 'Impact': [2.0,3.0,4.0]})

.. doctest::

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
	
    >>> coverage = chama.optimize.Coverage()
    >>> results = coverage.solve(sensor, scenario, det_times, 2)
	
P-median
-----------
The following P-median formulation is used to determine the optimal sensor
placement and type that minimizes impact (detection time or some other measure of damage):

.. math::
   
	\text{minimize} \qquad &\sum_{a \in A} \alpha_a \sum_{i \in {\cal L}_a} d_{ai} x_{ai}\\
	\text{subject to} \qquad &\sum_{i\in {\cal L}_a} x_{ai} = 1 \hspace{1.2in}      \forall a \in A\\ 
	&x_{ai} \le s_i       \hspace{1.47in}      \forall a \in A, i \in {\cal L}_a\\  
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

The P-median formulation is written in Pyomo [HLWW12]_ and solved
using open source or commercial solvers.  The open source GLPK solver
[Makh10]_ is used by default.  

The user supplies the impact coefficients, :math:`d_{ai}`, sensor budget, :math:`p`,
and (optionally) sensor cost, :math:`c_i`, and (optionally) the scenario
probability, :math:`\alpha_a`.  
The impact coefficients are computed from
transport simulation results and sensor characteristics, as described in
the :ref:`impact` Section.  
If sensor cost is not defined, it is assumed to be 1 for each sensor
(in that case, the sensor budget is the number of sensors to place).
If scenario probability is not defined, it is assumed to be equal for 
all scenarios.

The size of the optimization problem is determined by the number of
binary variables.  Although :math:`x_{ai}` is a binary indicator
variable, it is relaxed to be continuous between 0 and 1, and yet it
always converges to a value of 0 or 1. Therefore, the number of binary
variables that need to be considered by the solver is a function of the
number of candidate sensors alone, and not the number of scenarios
considered.  This formulation has been used to place sensors in large
water distribution networks [USEPA12]_ and [USEPA15]_.

The following example...

.. doctest::

    >>> pmedian = chama.optimize.Pmedian()
    >>> results = pmedian.solve(sensor, scenario, min_det_time, 2)
