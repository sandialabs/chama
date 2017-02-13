.. raw:: latex

    \newpage

Optimization
===========================

The general sensor placement optimization formulation is described below:

.. math::

	\text{minimize} \qquad & \sum_{a \in A} \alpha_a \sum_{i \in {\cal L}_a} d_{ai} x_{ai} \label{eqn:eSP} \\
	\text{subject to} \qquad &\sum_{i\in {\cal L}_a} x_{ai} = 1 &&\forall a \in A\\ 
	&x_{ai} \le s_i &&\forall a \in A, i \in {\cal L}_a\\  
	&\sum_{i \in L} c_i s_i \le p\\ 
	&s_i \in \{0,1\} &&\forall i \in L\\ 
	&0 \leq x_{ai} \leq 1 &&\forall a \in A, i \in {\cal L}_a 

where 
:math:`A` is the set of all leak scenarios, 
:math:`L` is the set of all candidate sensor locations, and 
:math:`L_a` is the set of all sensor locations that are capable of detecting leak scenario :math:`a`. 
The parameter :math:`D_{ai}` is the damage coefficient, and represents some measure of the damage that will be incurred if the leak scenario is first detected by a sensor at location :math:`i`.  
The variable :math:`x_{ai}` is an indicator variable that will be 1 if there is a sensor installed at location :math:`i` and that sensor is the first to detect leak scenario a (where “first” is defined as the minimum possible damage, usually defined as time to detection). 
Each scenario can be assigned a probability of occurrence, :math:`\alpha`.  
The variables :math:`s_i` is a binary variable that will be 1 if location i is selected, and 0 otherwise. 
The parameter :math:`p` is the total number of sensors that are allowed in the optimal placement. 

This formulation has been used to place sensors in large water networks [USEPA12]_ and [USEPA15]_.  
It is remarkably efficient for large problems since, although :math:`x_{ai}` is binary 
indicator variable, it can be relaxed to be a continuous variable between 0 and 
1, and yet it will always converge to a value of 0 or 1. Therefore, the number 
of binary variables that need to be considered by the solver is a function of 
the number of locations alone, and not the number of leak scenarios considered. 
