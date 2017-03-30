.. raw:: latex

    \newpage

Optimization
===========================

The following formulation is used to determine the optimal sensor placement and type that minimizes damage given a set of scenarios.

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

* :math:`L_a` is the set of all sensors that are capable of detecting scenario :math:`a`

* :math:`\alpha` is the scenario probability of occurrence

* :math:`d_{ai}` is the damage coefficient, and represents some measure of the damage that will be incurred if scenario :math:`a` is first detected by sensor :math:`i`  

* :math:`x_{ai}` is an indicator variable that will be 1 if sensor :math:`i` is installed and that sensor is the first to detect scenario :math:`a` (where `first` is defined as the minimum possible damage, usually defined as time to detection)

* :math:`s_i` is a binary variable that will be 1 if sensor :math:`i` is selected, and 0 otherwise

* :math:`c_i` is the cost of sensor :math:`i` 

* :math:`p` is the sensors budget

The optimization formulation is written in Pyomo [HLWW12]_ and solved using open source or commercial solvers.  
The open source GLPK solver [Makh10]_ is used by default.
The :mod:`chama.solver` module has more information on setting up and running sensor placement optimization. 

The user supplies the damage coefficient, :math:`d_{ai}`, sensor cost, :math:`c_i`, sensor budget, :math:`p`, and (optionally) the scenario probability, :math:`\alpha`.
The damage coefficient is computed from transport simulation results and sensor characteristics, as described in the :ref:`impact` Section. 
To place N sensors, set the sensor budget to N and sensor cost to 1.0 for each sensor.
If scenario probability is not defined, it is assumed to be equal for all scenarios.

The size of the optimization problem is determined by the number of binary variables.  
Although :math:`x_{ai}` is a binary 
indicator variable, it is relaxed to be continuous between 0 and 
1, and yet it always converges to a value of 0 or 1. Therefore, the number 
of binary variables that need to be considered by the solver is a function of 
the number of candidate sensors alone, and not the number of scenarios considered. 
This formulation has been used to place sensors in large water distribution networks [USEPA12]_ and [USEPA15]_.  