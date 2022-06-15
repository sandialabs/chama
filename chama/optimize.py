"""
The optimize module contains high-level solvers for sensor placement
optimization.

.. rubric:: Contents

.. autosummary::

    ImpactFormulation
    CoverageFormulation
"""

from __future__ import print_function, division
import pyomo.environ as pe
import chama.utils as cu
import numpy as np
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import pyomo.environ as aml
import itertools

dummy_sensor_name = '__DUMMY_SENSOR_UNDETECTED__'

# ToDo: lookup how to reference a method in rst.


class ImpactFormulation(object):
    """
    Sensor placement based on minimizing average impact across a set of scenarios.
    """

    def __init__(self):
        self._model = None
        self._impact = None
        self._sensor = None
        self._scenario = None
        self._impact_col_name = 'Impact'
        self._use_scenario_probability = False
        self._use_sensor_cost = False
        self._solved = False

    def solve(self, impact=None, sensor=None, scenario=None,
              sensor_budget=None, use_sensor_cost=False,
              use_scenario_probability=False, impact_col_name='Impact',
              mip_solver_name='glpk', pyomo_options=None, solver_options=None):
        """
        Solves the sensor placement optimization by minimizing impact.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact assessment. Impact is stored as a pandas DataFrame with columns 
            **Scenario**, **Sensor**, and **Impact**. Each row contains a single 
            detection time (or other measure of impact/damage) for a sensor 
            that detects a scenario. The column name for Impact can also be 
            specified by the user using the argument 'impact_col_name'.
        sensor : pandas DataFrame
            Sensor characteristics. Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns **Sensor** and **Cost**. Cost is used in the sensor placement
            optimization if the 'use_sensor_cost' flag is set to True.
        scenario : pandas DataFrame
            Scenario characteristics. Contains scenario probability and the
            impact for undetected scenarios. Scenario characteristics are
            stored as a pandas DataFrame with columns **Scenario**,
            **Undetected Impact**, and **Probability**. Undetected Impact is
            required for each scenario. Probability is used if the
            'use_scenario_probability' flag is set to True.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the
            optimization. If False, sensors have equal cost of 1.
        use_scenario_probability : bool
            Boolean indicating if scenario probability should be used in the
            optimization. If False, scenarios have equal probability.
        impact_col_name : str
            The name of the column containing the impact data to be used
            in the objective function.
        mip_solver_name : str
            Optimization solver name passed to Pyomo. The solver must be
            supported by Pyomo and support solution of mixed-integer
            programming problems.
        pyomo_options : dict
            Keyword arguments to be passed to the Pyomo solver .solve method
            Defaults to an empty dictionary.
        solver_options : dict
            Solver specific options to pass through Pyomo to the underlying solver.
            Defaults to an empty dictionary.

        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean impact based on the selected sensors
            * FractionDetected: The fraction of scenarios that were detected
            * TotalSensorCost: Total cost of the selected sensors
            * Assessment: The impact value for each sensor-scenario pair.
              The assessment is stored as a pandas DataFrame with columns
              **Scenario**, **Sensor**, and **Impact** (same format as the input
              Impact assessment). If the selected sensors did not detect a
              particular scenario, the impact is set to the Undetected Impact.
        """

        self.create_pyomo_model(impact=impact, sensor=sensor, scenario=scenario,
                                sensor_budget=sensor_budget, use_sensor_cost=use_sensor_cost,
                                use_scenario_probability=use_scenario_probability,
                                impact_col_name=impact_col_name)

        self.solve_pyomo_model(sensor_budget=sensor_budget, mip_solver_name=mip_solver_name,
                               pyomo_options=pyomo_options, solver_options=solver_options)

        results_dict = self.create_solution_summary()

        return results_dict

    def create_pyomo_model(self, impact=None, sensor=None, scenario=None, sensor_budget=None,
                           use_sensor_cost=False, use_scenario_probability=False,
                           impact_col_name='Impact'):
        """
        Returns the Pyomo model. 
        
        See :py:meth:`ImpactFormulation.solve` for more information on parameters.
        
        Returns
        -------
        Pyomo ConcreteModel ready to be solved
        """
        # reset the internal model and data attributes
        # BLN: Why do we reset these when they will be overwritten at the
        # end of this method anyway?
        self._model = None
        self._impact = None
        self._sensor = None
        self._scenario = None

        # validate the pandas dataframe input
        cu._df_columns_required('impact', impact,
                                {'Scenario': object,
                                 'Sensor': object,
                                 impact_col_name: [float, int]}) 
        cu._df_nans_not_allowed('impact', impact)
        if sensor is not None:
            cu._df_columns_required('sensor', sensor,
                                   {'Sensor': object})
            cu._df_nans_not_allowed('sensor', sensor)

            sensor = sensor.set_index('Sensor')
            assert(sensor.index.names[0] == 'Sensor')

        cu._df_columns_required('scenario', scenario,
                               {'Scenario': object,
                               'Undetected Impact': [float, int]}) 
        cu._df_nans_not_allowed('scenario', scenario)

        # validate optional columns in pandas dataframe input
        if use_scenario_probability:
            cu._df_columns_required('scenario', scenario,
                                    {'Probability': float})

        if use_sensor_cost:
            if sensor is None:
                raise ValueError(
                    'ImpactFormulation: use_sensor_cost cannot be '
                    'True if "sensor" DataFrame is not provided.')
            cu._df_columns_required('sensor', sensor,
                                    {'Cost': [float, int]}) 

        # Notice, setting the index here
        impact = impact.set_index(['Scenario', 'Sensor'])
        assert(impact.index.names[0] == 'Scenario')
        assert(impact.index.names[1] == 'Sensor')

        # Python set will extract the unique Scenario and Sensor values
        scenario_list = sorted(scenario['Scenario'].unique())

        # Always get sensor list from impact DataFrame in case there are
        # sensors in the sensor DataFrame that didn't detect anything and
        # therefore do not appear in the impact DataFrame
        sensor_list = sorted(set(impact.index.get_level_values('Sensor')))

        if use_sensor_cost:
            sensor_cost = sensor['Cost']
        else:
            sensor_cost = pd.Series(data=[1.0]*len(sensor_list), index=sensor_list)

        # Add in the data for the dummy sensor to account for a scenario that
        # is undetected
        sensor_list.append(dummy_sensor_name)

        df_dummy = pd.DataFrame(scenario_list, columns=['Scenario'])
        df_dummy = df_dummy.set_index(['Scenario'])

        scenario = scenario.set_index(['Scenario'])
        df_dummy[impact_col_name] = scenario['Undetected Impact']
        scenario.reset_index(level=[0], inplace=True)

        df_dummy['Sensor'] = dummy_sensor_name
        df_dummy = df_dummy.reset_index().set_index(['Scenario', 'Sensor'])
        impact = pd.concat([impact, df_dummy])
        sensor_cost[dummy_sensor_name] = 0.0

        # Create a list of tuples for all the scenario/sensor pairs where
        # detection has occurred
        scenario_sensor_pairs = impact.index.tolist()

        # Create the (jagged) index set of sensors that were able to detect a
        # particular scenario
        scenario_sensors = dict()
        for (a, i) in scenario_sensor_pairs:
            if a not in scenario_sensors:
                scenario_sensors[a] = list()
            scenario_sensors[a].append(i)

        # create the model container
        model = pe.ConcreteModel()
        model.scenario_sensors = scenario_sensors

        # Pyomo does not create an ordered dummy set when passed a list - do
        # this for now as a workaround
        model.scenario_set = pe.Set(initialize=scenario_list, ordered=True)
        model.sensor_set = pe.Set(initialize=sensor_list, ordered=True)
        model.scenario_sensor_pairs_set = \
            pe.Set(initialize=scenario_sensor_pairs, ordered=True)

        # create mutable parameter that may be changed
        model.sensor_budget = pe.Param(initialize=sensor_budget, mutable=True, within=aml.Any)

        # x_{a,i} variable indicates which sensor is the first to detect
        # scenario a
        model.x = pe.Var(model.scenario_sensor_pairs_set, bounds=(0, 1))

        # y_i variable indicates if a sensor is installed or not
        model.y = pe.Var(model.sensor_set, within=pe.Binary)

        # objective function minimize the sum impact across all scenarios
        if use_scenario_probability:
            scenario.set_index(['Scenario'], inplace=True)
            model.obj = pe.Objective(expr= \
                sum(float(scenario.at[a, 'Probability']) *
                float(impact[impact_col_name].loc[a, i]) * model.x[a, i]
                for (a, i) in scenario_sensor_pairs))
        else:
            model.obj = pe.Objective(expr= \
                1.0 / float(len(scenario_list)) *
                sum(float(impact[impact_col_name].loc[a, i]) * model.x[a, i]
                for (a, i) in scenario_sensor_pairs))

        # constrain the problem to have only one x value for each scenario
        def limit_x_rule(m, a):
            return sum(m.x[a, i] for i in scenario_sensors[a]) == 1
        model.limit_x = pe.Constraint(model.scenario_set, rule=limit_x_rule)

        # can only detect scenario a with location i if location i is selected
        def detect_only_if_sensor_rule(m, a, i):
            return m.x[a, i] <= model.y[i]
        model.detect_only_if_sensor = \
            pe.Constraint(model.scenario_sensor_pairs_set,
                          rule=detect_only_if_sensor_rule)

        # limit the number of sensors
        model.total_sensor_cost = pe.Expression(expr=sum(float(sensor_cost[i]) * model.y[i] for i in sensor_list))
        model.sensor_budget_con = pe.Constraint(expr= model.total_sensor_cost <= model.sensor_budget)

        self._model = model
        impact.reset_index(inplace=True)
        self._impact = impact
        self._sensor = sensor
        scenario.reset_index(inplace=True)
        self._scenario = scenario
        self._impact_col_name = impact_col_name
        self._use_sensor_cost = use_sensor_cost
        self._use_scenario_probability = use_scenario_probability

        # Any changes to the model require re-solving
        self._solved = False

        return model

    def add_grouping_constraint(self, sensor_list, select=None,
                                min_select=None, max_select=None):
        """
        Adds a sensor grouping constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

        The keyword argument 'select' enforces an equality constraint,
        while 'min_select' and 'max_select' correspond to lower and upper
        bounds on the grouping constraints, respectively. You can specify
        one or both of 'min_select' and 'max_select' OR use 'select'

        Parameters
        ----------
        sensor_list : list of strings
            List containing the string names of a subset of the sensors
        select : positive integer or None
            The exact number of sensors from the sensor_list that should
            be selected
        min_select : positive integer or None
            The minimum number of sensors from the sensor_list that should
            be selected
        max_select : positive integer or None
            The maximum number of sensors from the sensor_list that should
            be selected
        """
        _add_grouping_constraint(self, sensor_list=sensor_list, select=select, 
                                 min_select=min_select, max_select=max_select)

    def solve_pyomo_model(self, sensor_budget=None, mip_solver_name='glpk',
                          pyomo_options=None, solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        See :py:meth:`ImpactFormulation.solve` for more information on parameters.
        """
        if self._model is None:
            raise RuntimeError('Cannot call solve_pyomo_model before the model'
                               ' is created with create_pyomo_model'
                               )

        # change the sensor budget if necessary
        if sensor_budget is not None:
            self._model.sensor_budget = sensor_budget

        (solved, results) = _solve_pyomo_model(self._model, mip_solver_name=mip_solver_name, pyomo_options=pyomo_options,
                                     solver_options=solver_options)
        self._solved = solved

    def create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        See :py:meth:`ImpactFormulation.solve` for more information on the solution summary.
        
        Returns
        -------
        Dictionary containing a summary of results.
        """
		
        if self._model is None:
            raise RuntimeError('Cannot call create_solution_summary before '
                               'the model is created and solved.'
                               )
        if not self._solved:
            return {'Solved': self._solved,
                    'Objective': None,
                    'Sensors': None,
                    'ImpactAssessment': None}

        model = self._model
        impact_df = self._impact
        scenario_df = self._scenario
        selected_sensors = []
        for key in model.y:
            if pe.value(model.y[key]) > 0.5:
                if key != dummy_sensor_name:
                    selected_sensors.append(key)

        obj_value = pe.value(model.obj)
        selected_impact = {'Scenario': [], 'Sensor': [], 'Impact': []}
        for key in model.x:
            scenario = key[0]
            sensor = key[1]
            if pe.value(model.x[(scenario, sensor)]) > 0.5:
                if sensor == dummy_sensor_name:
                    sensor = None
                    impact_val = \
                        scenario_df[scenario_df['Scenario'] ==
                                    scenario]['Undetected Impact'].values[0]
                else:
                    
                    impact_val = \
                        impact_df[(impact_df['Scenario'] == scenario) &
                                  (impact_df['Sensor'] == sensor)][
                            self._impact_col_name].values[0]
                selected_impact['Scenario'].append(scenario)
                selected_impact['Sensor'].append(sensor)
                selected_impact['Impact'].append(impact_val)

        selected_impact = pd.DataFrame(selected_impact)
        selected_impact = selected_impact[['Scenario', 'Sensor', 'Impact']]

        frac_detected = 0
        for a in model.scenario_set:
            detected = sum(pe.value(model.x[a,i]) for i in model.scenario_sensors[a] if i != dummy_sensor_name)
            eps = 1e-6
            assert(detected >= 0-eps and detected <= 1+eps)
            if detected > 0.5:
                frac_detected += 1
        frac_detected = float(frac_detected)/float(len(model.scenario_set))

        return {'Solved': self._solved,
                 'Objective': obj_value,
                 'Sensors': selected_sensors,
                 'FractionDetected': frac_detected,
                 'TotalSensorCost': pe.value(model.total_sensor_cost),
                 'Assessment': selected_impact}


class CoverageFormulation(object):
    """
    Sensor placement based on maximizing coverage of a set of entities.
    An 'entity' can represent geographic areas, scenarios, or scenario-time pairs.
    """
    def __init__(self):
        self._model = None

    def solve(self, coverage, sensor=None, entity=None, sensor_budget=None,
              use_sensor_cost=None, use_entity_weight=False, redundancy=0, coverage_col_name='Coverage',
              mip_solver_name='glpk', pyomo_options=None, solver_options=None):
        """
        Solves the sensor placement optimization by maximizing coverage.

        Parameters
        ----------
        coverage : pandas DataFrame
            Coverage data. Coverage is stored as a pandas DataFrame with columns
            **Sensor** and **Coverage**.  Each row contains a list of entities 
            that are covered by single sensor. The column name for Coverage can also be 
            specified by the user using the argument 'coverage_col_name'.
        sensor : pandas DataFrame
            Sensor characteristics. Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns **Sensor** and **Cost**. Cost is used in the sensor placement
            optimization if the 'use_sensor_cost' flag is set to True.
        entity : pandas DataFrame
            Entity characteristics. Contains entity weights. 
            Entity characteristics are stored as a pandas DataFrame with columns 
            **Entity** and **Weight**. Weight is used if the
            'use_entity_weight' flag is set to True.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the
            optimization. If False, sensors have equal cost of 1.
        use_entity_weight : bool
            Boolean indicating if entity weights should be used in the
            optimization. If False, entities have equal weight.
        redundancy : int
            Redundancy level. A value of 0 means only one sensor is required to 
            covered an entity, whereas a value of 1 means two sensors must 
            cover an entity before it considered covered.
        coverage_col_name : str
            The name of the column containing the coverage data to be used
            in the objective function.
        mip_solver_name : str
            Optimization solver name passed to Pyomo. The solver must be
            supported by Pyomo and support solution of mixed-integer
            programming problems.
        pyomo_options : dict
            Keyword arguments to be passed to the Pyomo solver .solve method.
            Defaults to an empty dictionary.
        solver_options : dict
            Solver specific options to pass through Pyomo to the underlying solver.
            Defaults to an empty dictionary.
            
        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean coverage based on the selected sensors
            * FractionDetected: the fraction of entities that are detected
            * TotalSensorCost: Total cost of the selected sensors
            * EntityAssessment: a dictionary whose keys are the entity names, 
              and values are a list of sensors that detect that entity
            * SensorAssessment: a dictionary whose keys are the sensor names, 
              and values are the list of entities that are detected by that sensor
        """
		
        self.create_pyomo_model(coverage=coverage, sensor=sensor, entity=entity,
                                sensor_budget=sensor_budget, use_sensor_cost=use_sensor_cost,
                                use_entity_weight=use_entity_weight, redundancy=redundancy,
                                coverage_col_name=coverage_col_name)

        self.solve_pyomo_model(sensor_budget=sensor_budget, mip_solver_name=mip_solver_name,
                               pyomo_options=pyomo_options, solver_options=solver_options)

        # might want to throw this exception, might want to pass this through to the results object
        if not self._model.solved:
            raise RuntimeError("Optimization failed to solve. Please set pyomo_options={'tee': True}"
                               " and check solver logs.")

        results_dict = self.create_solution_summary()

        return results_dict


    def create_pyomo_model(self, coverage, sensor=None, entity=None, sensor_budget=None, use_sensor_cost=False,
                           use_entity_weight=False, redundancy=0, coverage_col_name='Coverage'):
        """
        Returns the Pyomo model. 
        
        See :py:meth:`CoverageFormulation.solve` for more information on parameters.
        
        Returns
        -------
        Pyomo ConcreteModel ready to be solved
        """
		
        self._model = None

        self._model= model = pe.ConcreteModel()

        # build the list of entities from the coverage DataFrame
        covered_items = coverage['Coverage'].tolist()
        entity_list = cu._unique_items_from_list_of_lists(covered_items)
        if entity is None:
            if use_entity_weight:
                raise ValueError('CoverageFormulation: use_entity_weight cannot be True if'
                                 '"entity" DataFrame is not provided.')
        else:
            # add potential additional entities from the entity DataFrame
            additional_entities = set(entity['Entity'].unique())
            entity_list = entity_list.union(additional_entities)
        
        entity_list = sorted(entity_list)
    
        # TODO: Add DataFrame column checks like in the ImpactFormulation

        if sensor is None and use_sensor_cost:
            raise ValueError('CoverageFormulation: use_sensor_cost cannot be True if "sensor" DataFrame is not provided.')

        # Always get sensor list from coverage DataFrame in case there are
        # sensors in the sensor DataFrame that didn't detect anything and
        # therefore do not appear in the coverage DataFrame
        sensor_list = sorted(coverage['Sensor'].unique())

        # make a series of the coverage column (for faster access)
        coverage_series = coverage.set_index('Sensor')[coverage_col_name]

        # create a dictionary of sets where the key is the entity, and the
        # value is the set of sensors that covers that entity
        entity_sensors = {e:set() for e in entity_list}
        for s in sensor_list:
            s_entities = coverage_series[s]

            for e in s_entities:
                entity_sensors[e].add(s)

        for e in entity_sensors.keys():
            entity_sensors[e] = list(sorted(entity_sensors[e]))

        model.entity_list = pe.Set(initialize=entity_list, ordered=True)
        model.sensor_list = pe.Set(initialize=sensor_list, ordered=True)

        if redundancy > 0:
            model.x = pe.Var(model.entity_list, within=pe.Binary)
        else:
            model.x = pe.Var(model.entity_list, bounds=(0,1))
        model.y = pe.Var(model.sensor_list, within=pe.Binary)

        if use_entity_weight:
            entity_weights = entity.set_index('Entity')['Weight']
            # Check for missing entity weights, set to 0
            missing_entities = set(entity_list) - set(entity_weights.index)
            assert len(missing_entities) == 0, "Missing entity weights , " +  str(missing_entities)
            model.obj = pe.Objective(expr=sum(float(entity_weights[e])*model.x[e] for e in entity_list), sense=pe.maximize)
        else:
            model.obj = pe.Objective(expr=sum(model.x[e] for e in entity_list), sense=pe.maximize)

        def entity_covered_rule(m, e):
            if redundancy > 0:
                return (redundancy + 1.0)*m.x[e] <= sum(m.y[b] for b in entity_sensors[e])
            return m.x[e] <= sum(m.y[b] for b in entity_sensors[e])
        model.entity_covered = pe.Constraint(model.entity_list, rule=entity_covered_rule)

        if sensor_budget is None:
            if use_sensor_cost:
                raise ValueError('CoverageFormulation: sensor_budget must be specified if use_sensor_cost is set to True.')
            sensor_budget = len(sensor_list) # no sensor budget provided - allow all sensors
        model.sensor_budget = pe.Param(initialize=sensor_budget, mutable=True, within=aml.Any)

        if use_sensor_cost:
            sensor_cost = sensor.set_index('Sensor')['Cost']
            model.total_sensor_cost = pe.Expression(expr=sum(sensor_cost[s]*model.y[s] for s in sensor_list))
        else:
            model.total_sensor_cost = pe.Expression(expr=sum(model.y[s] for s in sensor_list))
        model.sensor_upper_limit = pe.Constraint(expr= model.total_sensor_cost <= model.sensor_budget)

        model.entity_sensors = entity_sensors
        model.solved = False
        self._model = model
        return model
    
    def add_grouping_constraint(self, sensor_list, select=None,
                                min_select=None, max_select=None):
        """
        Adds a sensor grouping constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

        The keyword argument 'select' enforces an equality constraint,
        while 'min_select' and 'max_select' correspond to lower and upper
        bounds on the grouping constraints, respectively. You can specify
        one or both of 'min_select' and 'max_select' OR use 'select'

        Parameters
        ----------
        sensor_list : list of strings
            List containing the string names of a subset of the sensors
        select : positive integer or None
            The exact number of sensors from the sensor_list that should
            be selected
        min_select : positive integer or None
            The minimum number of sensors from the sensor_list that should
            be selected
        max_select : positive integer or None
            The maximum number of sensors from the sensor_list that should
            be selected
        """
        _add_grouping_constraint(self, sensor_list=sensor_list, select=select, 
                                 min_select=min_select, max_select=max_select)
        
        
    def solve_pyomo_model(self, sensor_budget=None, mip_solver_name='glpk',
                          pyomo_options=None, solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        See :py:meth:`CoverageFormulation.solve` for more information on parameters.
        """

        if self._model is None:
            raise RuntimeError('Cannot call solve_pyomo_model before the model'
                               ' is created with create_pyomo_model'
                               )

        self._model.solved = False

        # change the sensor budget if necessary
        if sensor_budget is not None:
            self._model.sensor_budget = sensor_budget

        (solved, results) = _solve_pyomo_model(self._model, mip_solver_name=mip_solver_name,
                                               pyomo_options=pyomo_options, solver_options=solver_options)

        self._model.solved = solved

    def create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        See :py:meth:`CoverageFormulation.solve` for more information on the solution summary.
 
        Returns
        -------
        Dictionary containing a summary of results.
        """

        if self._model is None:
            raise RuntimeError('Cannot call create_solution_summary before '
                               'the model is created and solved.'
                               )

        model = self._model
        if not model.solved:
            return {'Solved': model.solved,
                    'Objective': None,
                    'Sensors': None,
                    'FractionDetected': None,
                    'EntityAssessment': None,
                    'SensorAssessment': None}

        selected_sensors = []
        for key in model.y:
            if pe.value(model.y[key]) > 0.5:
                selected_sensors.append(key)

        obj_value = pe.value(model.obj)

        frac_detected = sum(pe.value(model.x[e]) for e in model.x)/(len(model.x))

        entity_assessment = {e:[] for e in model.entity_list}
        for e in model.entity_list:
            for s in model.entity_sensors[e]:
                if pe.value(model.y[s]) > 0.5:
                    entity_assessment[e].append(s)

        sensor_assessment = dict()
        for s in model.sensor_list:
            if pe.value(model.y[s]) > 0.5:
                sensor_assessment[s] = [e for e in model.entity_list if s in model.entity_sensors[e]]

        return {'Solved': model.solved,
                'Objective': obj_value,
                'Sensors': selected_sensors,
                'FractionDetected': frac_detected,
                'TotalSensorCost': pe.value(model.total_sensor_cost),
                'EntityAssessment': entity_assessment,
                'SensorAssessment': sensor_assessment}

def _add_grouping_constraint(self, sensor_list, select=None,
                                min_select=None, max_select=None):
       
        #TODO: Should we make this easier by just allowing lower bound and
        #upper bound and do an equality if they are the same?
        if self._model is None:
            raise RuntimeError('Cannot add a grouping constraint to a'
                               'nonexistent model. Please call the '
                               'create_pyomo_model function before trying to '
                               'add grouping constraints')

        if select is not None and min_select is not None:
            raise ValueError('Invalid keyword arguments for adding grouping '
                             'constraint. Cannot specify both a "select" '
                             'value and a "min_select" value')

        if select is not None and max_select is not None:
            raise ValueError('Invalid keyword arguments for adding grouping '
                             'constraint. Cannot specify both a "select" '
                             'value and a "max_select" value')

        if select is None and max_select is None and min_select is None:
            raise ValueError('Must specify a sensor selection limit for the '
                             'grouping constraint.')

        gconlist = self._model.find_component('_groupingconlist')
        if gconlist is None:
            self._model._groupingconlist = pe.ConstraintList()
            gconlist = self._model._groupingconlist

        # Check to make sure all sensors are valid and build sum expression
        sensor_sum = sum(self._model.y[i] for i in sensor_list)

        if select is not None:
            #  Select exactly 'select' sensors from sensor_list
            if select < 0:
                raise ValueError('Cannot select a negative number of sensors')

            gconlist.add(sensor_sum == select)

        elif min_select is not None and max_select is not None:
            #  Select between min_select and max_select sensors from
            #  sensor_list
            if (min_select < 0) or (max_select < 0):
                raise ValueError('Cannot select a negative number of sensors')

            if min_select > max_select:
                raise ValueError('min_select must be less than max_select')

            #gconlist.add(min_select <= sensor_sum <= max_select) # Chained inequalities are deprecated
            gconlist.add(aml.inequality(min_select, sensor_sum, max_select))

        elif min_select is not None:
            #  Select at least min_select sensors from sensor list
            if min_select < 0:
                raise ValueError('Cannot select a negative number of sensors')
            gconlist.add(min_select <= sensor_sum)
        else:
            #  Select at most max_select sensors from sensor list
            if max_select < 0:
                raise ValueError('Cannot select a negative number of sensors')
            gconlist.add(sensor_sum <= max_select)

        # Any changes to the model require re-solving
        self._solved = False
        
def _solve_pyomo_model(model, mip_solver_name='glpk', pyomo_options=None, solver_options=None):
    """
    Internal method to solve the Pyomo model and check the optimization status
    """
    if pyomo_options is None:
        pyomo_options = {}

    if solver_options is None:
        solver_options = {}

    if model is None:
        raise RuntimeError('Cannot call solve_pyomo_model before the model'
                           ' is created with create_pyomo_model'
                           )

    # create the solver
    opt = pe.SolverFactory(mip_solver_name)

    results = opt.solve(model, options=solver_options, **pyomo_options)

    # Check solver status
    solved = None
    if (results.solver.status == SolverStatus.ok) and \
            (results.solver.termination_condition == TerminationCondition.optimal):
        solved = True
    else:
        solved = False
        print('The solver was unable to find an optimal solution')

    return (solved, results)

