"""
The optimize module contains high-level solvers for sensor placement
optimization.

Includes the following strategies:
* ImpactSolver : Perform sensor placement based on minimizing the average impact across a set of scenarios.
* CoverageSolver : Perform sensor placement based on maximizing coverage of a set of entities (or minimizing
  the number of sensors required for a particular level of coverage).
* ScenarioCoverageSolver : A wrapper around the CoverageSolver that provides an intuitive interface for
  performing sensor placement based on coverage over a set of scenarios.
* GeographicCoverageSolver : A wrapper around the CoverageSolver that presents an interface for performing
  sensor placement based on coverage of a set of geographic cubes or grids
"""

from __future__ import print_function, division
import pyomo.environ as pe
import chama.utils as cu
import numpy as np
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition
import itertools

dummy_sensor_name = '__DUMMY_SENSOR_UNDETECTED__'

# ToDo: lookup how to reference a method in rst.

class BasePyomoSolver(object):
    def __init__(self):
        self._model = None
        self._solved = False

    def _add_subset_selection_constraint(self, sensor_var, sensor_list, lower_bound=None, upper_bound=None):
        """
        Adds a subset selection constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

        The constraint will allow us to require at least (lower bound) and no more than (upper bound) number of
        sensors from the sensor_list. Either lower_bound or upper_bound can be set to None, or both can be set
        (to set a range including both lower and upper bounds).

        Parameters
        ----------
        sensor_var : Pyomo Var
            The variable on the Pyomo model that should be included in the constraint (the sensor yes/no variable)
        sensor_list : list of strings
            List containing the string names of the subset of the sensors to be included in the subset selection
            constraint
        lower_bound : int or None
            Lower bound on the number of sensors that must be selected from sensor_list
        upper_bound : int or None
            Upper bound on the number of sensors that must be selected from sensor_list
        """
        # ensure the incoming bounds are None or integer - if not, exception will be raised
        if lower_bound is not None:
            lower_bound = int(lower_bound)
        if upper_bound is not None:
            upper_bound = int(upper_bound)
        if lower_bound is None and upper_bound is None:
            raise ValueError('At least one of upper_bound or lower_bound must be set when adding subset'
                             ' selection constraint.')

        # check that model is not None
        if self._model is None:
            raise ValueError('Cannot add a subset selection constraint to a'
                               'nonexistent model. Please ensure a valid Pyomo model '
                               'is created before trying to add additional constraints.'
                               )

        # find the ConstraintList used to store these constraints or build a new one if required
        subset_con_list = self._model.find_component('_subset_con_list')
        if subset_con_list is None:
            self._model._subset_con_list = pe.ConstraintList()
            subset_con_list = self._model._subset_con_list

        # Check to make sure all sensors are valid and build sum expression
        sensor_sum = sum(sensor_var[k] for k in sensor_list)

        # add the appropriate constraint
        if lower_bound == upper_bound:
            # bounds are equal - add equality constraint
            subset_con_list.add(sensor_sum == lower_bound)
        else:
            if lower_bound is not None:
                subset_con_list.add(sensor_sum >= lower_bound)
            if upper_bound is not None:
                subset_con_list.add(sensor_sum <= upper_bound)

        # Any changes to the model require re-solving
        # ToDo: discuss this object model - this is done differently in ImpactSolver and CoverageSolver
        self._solved = False

    def _add_integer_cut_from_model_solution(self, binary_var):
        """
        Adds a constraint to ensure that the current solution will not be valid in future solves,
        forcing the optimization to find the 'next best' solution.

        Parameters
        ----------
        model : Pyomo model
            The Pyomo model where the constraint should be added

        binary_var : Pyomo Var
            The Pyomo Var object (indexed) to use in the constraint. The current 0-1 values in

        Returns
        -------

        """
        key_list_0 = []
        key_list_1 = []

        for k in binary_var:
            if pe.value(binary_var[k]) < 0.5:
                key_list_0.append(k)
            else:
                key_list_1.append(k)

        # find the ConstraintList used to store these constraints or build a new one if required
        integer_cut_con_list = self._model.find_component('_integer_cut_con_list')
        if integer_cut_con_list is None:
            self._model._integer_cut_con_list = pe.ConstraintList()
            integer_cut_con_list = self._model._integer_cut_con_list
        integer_cut_con_list.add(expr= sum(binary_var[k] for k in key_list_0)
                                       - sum( (1.0-binary_var[k]) for k in key_list_1 ) >= 1)

        self._solved = False


class ImpactSolver(BasePyomoSolver):
    """
    Sensor placement based on minimizing average impact of a set of scenarios. Uses Pyomo to build and solve
    the optimization problem. See :py:meth:ImpactSolver.solve for usage details.
    """

    def __init__(self):
        super(ImpactSolver, self).__init__()
        self._impact = None
        self._sensor = None
        self._scenario = None
        self._use_scenario_probability = False
        self._use_sensor_cost = False

    def solve(self, impact=None, sensor=None, scenario=None,
              sensor_budget=None, use_sensor_cost=False,
              use_scenario_probability=False, impact_col_name='Impact',
              mip_solver_name='glpk', pyomo_options=None, solver_options=None):
        """
        Solves the sensor placement optimization.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact assessment. A single detection time (or other measure
            of damage) for each sensor that detects a scenario.
            Impact is stored as a pandas DataFrmae with columns 'Scenario',
            'Sensor', 'Impact'.
        sensor : pandas DataFrame
            Sensor characteristics. Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns 'Sensor' and 'Cost'. Cost is used in the sensor placement
            optimization if the 'use_sensor_cost' flag is set to True.
        scenario : pandas DataFrame
            Scenario characteristics.  Contains scenario probability and the
            impact for undetected scenarios. Scenario characteristics are
            stored as a pandas DataFrame with columns 'Scenario',
            'Undetected Impact', and 'Probability'. Undetected Impact is
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
            * FractionDetected: The fraction of the number of scenarios that were detected
            * TotalSensorCost: Total cost of the selected sensors
            * Assessment: The impact value for each sensor-scenario pair.
              The assessment is stored as a pandas DataFrame with columns
              'Scenario', 'Sensor', and 'Impact' (same format as the input
              Impact assessment) If the selected sensors did not detect a
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
        Returns the Pyomo model. See :py:meth:`Pmedian.solve` for more
        information on arguments.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact data
        sensor : pandas DataFrame
            Sensor characteristics
        scenario : pandas DataFrame
            Scenario characteristics
        sensor_budget : float
            Sensor budget
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used. Defaults to
            False, meaning sensors have equal cost of 1.
        use_scenario_probability : bool
            Boolean indicating if scenario probability should be used.
            Defaults to False, meaning scenarios have equal probability.
        impact_col_name : str
            The name of the column containing the impact data to be used
            in the objective function.

        Returns
        -------
        Pyomo ConcreteModel ready to be solved
        """
        # validate the pandas dataframe input
        cu._df_columns_required('impact', impact,
                                {'Scenario': np.object,
                                 'Sensor': np.object,
                                 impact_col_name: [np.float64, np.int64]})
        cu._df_nans_not_allowed('impact', impact)
        if sensor is not None:
            cu._df_columns_required('sensor', sensor,
                                   {'Sensor': np.object})
            cu._df_nans_not_allowed('sensor', sensor)

            sensor = sensor.set_index('Sensor')
            assert(sensor.index.names[0] == 'Sensor')

        cu._df_columns_required('scenario', scenario,
                               {'Scenario': np.object,
                               'Undetected Impact': [np.float64, np.int64]})
        cu._df_nans_not_allowed('scenario', scenario)

        # validate optional columns in pandas dataframe input
        if use_scenario_probability:
            cu._df_columns_required('scenario', scenario,
                                    {'Probability': np.float64})

        if use_sensor_cost:
            if sensor is None:
                raise ValueError(
                    'ImpactSolver formulation: use_sensor_cost cannot be '
                    'True if "sensor" DataFrame is not provided.')
            cu._df_columns_required('sensor', sensor,
                                    {'Cost': [np.float64, np.int64]})

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
        impact = impact.append(df_dummy)
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
        model.sensor_budget = pe.Param(initialize=sensor_budget, mutable=True)

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
        self._use_sensor_cost = use_sensor_cost
        self._use_scenario_probability = use_scenario_probability
        self._solved = False

        return model

    def add_subset_selection_constraint(self, sensor_list, lower_bound=None, upper_bound=None):
        """
        Adds a subset selection constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

        The constraint will allow us to require at least (lower bound) and no more than (upper bound) number of
        sensors from the sensor_list. Either lower_bound or upper_bound can be set to None, or both can be set
        (to set a range including both lower and upper bounds).

        Parameters
        ----------
        sensor_list : list of strings
            List containing the string names of the subset of the sensors to be included in the subset selection
            constraint
        lower_bound : int or None
            Lower bound on the number of sensors that must be selected from sensor_list
        upper_bound : int or None
            Upper bound on the number of sensors that must be selected from sensor_list
        """
        self._add_subset_selection_constraint(self._model.y, sensor_list, lower_bound, upper_bound)

    def solve_pyomo_model(self, sensor_budget=None, mip_solver_name='glpk',
                          pyomo_options=None, solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        See :py:meth:`ImpactSolver.solve` for more information on arguments.
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

    def add_integer_cut_from_model_solution(self):
        """
        Adds a constraint to ensure that the current solution will not be valid in future solves,
        forcing the optimization to find the 'next best' solution.
        """
        self._add_integer_cut_from_model_solution(self._model.y)

    def create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        Returns
        -------
        Dictionary containing objective value, selected sensors, and
        impact assessment.
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
                        (impact_df['Sensor'] == sensor)]['Impact'].values[0]
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


class CoverageSolver(BasePyomoSolver):
    def __init__(self):
        super(CoverageSolver,self).__init__()

    def solve(self, coverage, formulation='max-coverage', sensor=None, entities=None, sensor_budget=None,
              use_sensor_cost=None, use_entity_weights=False, redundancy=0, coverage_col_name='Coverage',
              mip_solver_name='glpk', pyomo_options=None, solver_options=None):
        """
        Solves the sensor placement optimization using the maximum coverage formulation

        Parameters
        ----------
        coverage : pandas DataFrame
            This DataFrame contains two columns. 'Sensor' is the name of the sensor, and the corresponding value in
            'Coverage', is the list of entities that are detected by that sensor.
        formulation : str
            Specifies the particular optimization formulation that should be used. The only supported
            formulation is 'max-coverage'.
        sensor : pandas DataFrame
            Sensor characteristics. Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns 'Sensor' and 'Cost'. This argument is only required if
            the 'use_sensor_cost' flag is set to True.
        entities : pandas DataFrame
            Characteristics of entities that should be covered (e.g., scenarios, times, geographical areas).
            DataFrame contains one or two columns. 'Entity' is the name of the entity. Optional 'Weight' is
            a weighting to use in the objective function to assign the value of covering each entity. This argument
            is only required if the 'use_entity_weights' flag is set to True.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the
            optimization. If False, sensors have equal cost of 1.
        use_entity_weights : bool
            Boolean indicating if entity weights should be used in the objective function.
            If False, each entity has equal probability.
        redundancy : int
            Redundancy level: A value of 0 means only one sensor is required to covered an entity, whereas
            a value of 1 means two sensors must cover an entity before it considered covered.
        coverage_col_name : str
            The name of the column containing the coverage data in the coverage DataFrame
        mip_solver_name : str
            Optimization solver name passed to Pyomo. The solver must be
            supported by Pyomo and support solution of mixed-integer
            programming problems.
        pyomo_options : dict
            Keyword arguments to be passed to the Pyomo solver .solve method
        solver_options : dict
            Solver specific options to pass through Pyomo to the underlying solver.
            Defaults to an empty dictionary.

        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean impact based on the selected sensors
            * FractionDetected: the fraction of all entities that are detected
            * EntityAssessment: a dictionary whose keys are the entity names, and values are a list of sensors
               that detect that entity
            * SensorAssessment: a dictionary whose keys are the sensor names, and values are the list of entities
              that are detected by that sensor

        """
        self.create_pyomo_model(coverage=coverage, sensor=sensor, entities=entities,
                                sensor_budget=sensor_budget, use_sensor_cost=use_sensor_cost,
                                use_entity_weights=use_entity_weights, redundancy=redundancy,
                                coverage_col_name=coverage_col_name)

        self.solve_pyomo_model(sensor_budget=sensor_budget, mip_solver_name=mip_solver_name,
                               pyomo_options=pyomo_options, solver_options=solver_options)

        # might want to throw this exception, might want to pass this through to the results object
        if not self._model.solved:
            raise RuntimeError("Optimization failed to solve. Please set pyomo_options={'tee': True}"
                               " and check solver logs.")

        results_dict = self.create_solution_summary()

        return results_dict

    def create_pyomo_model(self, coverage, sensor=None, entities=None, sensor_budget=None, use_sensor_cost=False,
                           use_entity_weights=False, redundancy=0, coverage_col_name='Coverage'):
        model = pe.ConcreteModel()

        entity_list = None
        if entities is None:
            if use_entity_weights:
                raise ValueError('CoverageSolver formulation: use_entity_weights cannot be True if'
                                 '"entities" DataFrame is not provided.')
            # build the list of entities from the coverage DataFrame
            covered_items = coverage['Coverage'].tolist()
            entity_list = sorted(cu.unique_items_from_list_of_lists(covered_items))
        else:
            entity_list = sorted(entities['Entity'].unique())

        # TODO: Add DataFrame column checks like in the ImpactSolver

        if sensor is None and use_sensor_cost:
            raise ValueError('CoverageSolver: use_sensor_cost cannot be True if "sensor" DataFrame is not provided.')

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

        if use_entity_weights:
            entity_weights = entities.set_index('Entity')['Weight']
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
                raise ValueError('CoverageSolver: sensor_budget must be specified if use_sensor_cost is set to True.')
            sensor_budget = len(sensor_list) # no sensor budget provided - allow all sensors
        model.sensor_budget = pe.Param(initialize=sensor_budget, mutable=True)

        if use_sensor_cost:
            sensor_cost = sensor.set_index('Sensor')['Cost']
            model.total_sensor_cost = pe.Expression(expr=sum(sensor_cost[s]*model.y[s] for s in sensor_list))
        else:
            model.total_sensor_cost = pe.Expression(expr=sum(model.y[s] for s in sensor_list))
        model.sensor_upper_limit = pe.Constraint(expr= model.total_sensor_cost <= model.sensor_budget)

        self._model = model
        self._model.entity_sensors = entity_sensors
        self._solved = False
        return model

    def add_subset_selection_constraint(self, sensor_list, lower_bound=None, upper_bound=None):
        """
        Adds a subset selection constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

        The constraint will allow us to require at least (lower bound) and no more than (upper bound) number of
        sensors from the sensor_list. Either lower_bound or upper_bound can be set to None, or both can be set
        (to set a range including both lower and upper bounds).

        Parameters
        ----------
        sensor_list : list of strings
            List containing the string names of the subset of the sensors to be included in the subset selection
            constraint
        lower_bound : int or None
            Lower bound on the number of sensors that must be selected from sensor_list
        upper_bound : int or None
            Upper bound on the number of sensors that must be selected from sensor_list
        """
        self._add_subset_selection_constraint(self._model.y, sensor_list, lower_bound, upper_bound)


    def solve_pyomo_model(self, sensor_budget=None, mip_solver_name='glpk',
                          pyomo_options=None, solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        See :py:meth:`CoverageSolver.solve` for more information on arguments.
        """
        if self._model is None:
            raise RuntimeError('Cannot call solve_pyomo_model before the model'
                               ' is created with create_pyomo_model'
                               )

        self._solved = False

        # change the sensor budget if necessary
        if sensor_budget is not None:
            self._model.sensor_budget = sensor_budget

        (solved, results) = _solve_pyomo_model(self._model, mip_solver_name=mip_solver_name,
                                               pyomo_options=pyomo_options, solver_options=solver_options)

        self._solved = solved

    def add_integer_cut_from_model_solution(self):
        """
        Adds a constraint to ensure that the current solution will not be valid in future solves,
        forcing the optimization to find the 'next best' solution.
        """
        self._add_integer_cut_from_model_solution(self._model.y)

    def create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        Returns
        -------
        Dictionary with the following keys:
        * 'Solved': True/False, indicates if the optimization problem solved sucessfully or not
        * 'Objective': the value of the objective function (meaning depends on options selected)
        * 'Sensors': the optimal selection of sensors
        * 'FractionDetected': the fraction of all entities that are detected
        * 'EntityAssessment': a dictionary whose keys are the entity names, and values are a list of sensors
           that detect that entity
        * 'SensorAssessment': a dictionary whose keys are the sensor names, and values are the list of entities
           that are detected by that sensor
        """

        if self._model is None:
            raise RuntimeError('Cannot call create_solution_summary before '
                               'the model is created and solved.'
                               )

        if not self._solved:
            return {'Solved': self._solved,
                    'Objective': None,
                    'Sensors': None,
                    'FractionDetected': None,
                    'EntityAssessment': None,
                    'SensorAssessment': None}
        model = self._model

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


class ScenarioCoverageSolver(CoverageSolver):
    def __init__(self):
        super(ScenarioCoverageSolver,self).__init__()

    def solve(self, coverage, formulation='max-coverage', sensor=None, scenario=None, sensor_budget=None,
              use_sensor_cost=False, use_scenario_probability=False, redundancy=0,
              coverage_col_name='Coverage', mip_solver_name='glpk', pyomo_options=None,
              solver_options=None):

        """
        Solves the sensor placement optimization using coverage.

        Parameters
        ----------
        coverage : pandas DataFrame
            This DataFrame contains two columns. 'Sensor' is the name of the sensor, and the corresponding value in
            'Coverage', is the list of scenario names that are detected by that sensor.
        sensor : pandas DataFrame
            Sensor characteristics.  Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns 'Sensor' and 'Cost'. Cost is used in the sensor
            placement optimization if the 'use_sensor_cost' flag is set to True.
        scenario : pandas DataFrame
            Scenario characteristics. A pandas DataFrame with columns 'Scenario' that
            provides the name of the scenarios, and 'Probability' that defines the
            scenario probability (or any other weights for the scenarios).
            Probability is only used if the
            'use_scenario_probability' flag is set to True.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the optimization.
            If False, sensors have equal cost of 1.
        use_scenario_probability : bool
            Boolean indicating if scenario probability should be used in the optimization.
            If False, scenarios have equal probability.
        coverage_col_name : str
            The name of the column in coverage containing the coverage data (list of scenario names detected).
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
            * FractionDetected: the fraction of all entities that are detected
            * EntityAssessment: a dictionary whose keys are the entity names, and values are a list of sensors
               that detect that entity
            * SensorAssessment: a dictionary whose keys are the sensor names, and values are the list of entities
              that are detected by that sensor
        """
        if scenario is not None:
            scenario.rename(columns={'Scenario':'Entity', 'Probability':'Weight'},inplace=True)

        return super(ScenarioCoverageSolver,self).solve(coverage, formulation=formulation, sensor=sensor, entities=scenario,
                                                 sensor_budget=sensor_budget, use_sensor_cost=use_sensor_cost,
                                                 use_entity_weights=use_scenario_probability, redundancy=redundancy,
                                                 coverage_col_name=coverage_col_name, mip_solver_name=mip_solver_name,
                                                 pyomo_options=pyomo_options, solver_options=solver_options)


class GeographicCoverageSolver(CoverageSolver):
    def __init__(self):
        super(GeographicCoverageSolver,self).__init__()

    def solve(self, coverage, sensor=None, geo_loc=None, sensor_budget=None,
              use_sensor_cost=False, use_geo_loc_weights=False,
              coverage_col_name='Coverage', mip_solver_name='glpk', pyomo_options=None,
              solver_options=None):
        """
        Solves the sensor placement optimization using coverage.

        Parameters
        ----------
        coverage : pandas DataFrame
            This DataFrame contains two columns. 'Sensor' is the name of the sensor, and the corresponding value in
            'Coverage', is the list of scenario names that are detected by that sensor.
        sensor : pandas DataFrame
            Sensor characteristics.  Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns 'Sensor' and 'Cost'. Cost is used in the sensor
            placement optimization if the 'use_sensor_cost' flag is set to True.
        geo_loc : pandas DataFrame
            Characteristics of the geographic areas. This DataFrame has two
            columns. 'Location' contains the names of the geographic locations, and
            'Weight' contains the weighting (priority) for each of the geographic
            locations. This DataFrame is only required (and 'Weight' is only used) if the
            'use_geo_loc_weights' flag is set to True.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the optimization.
            If False, sensors have equal cost of 1.
        use_geo_loc_weights : bool
            Boolean indicating if weighting priorities for the geographic locations
            (specified in the geo_loc DataFrame) should be used.
            If False, all locations have equal weight.
        coverage_col_name : str
            The name of the column in coverage containing the coverage data (list of scenario names detected).
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
            * FractionDetected: the fraction of all entities that are detected
            * EntityAssessment: a dictionary whose keys are the entity names, and values are a list of sensors
               that detect that entity
            * SensorAssessment: a dictionary whose keys are the sensor names, and values are the list of entities
              that are detected by that sensor
        """
        if geo_loc is not None:
            geo_loc.rename(columns={'Location':'Entity'}, inplace=True)

        return super(ScenarioCoverageSolver, self).solve(coverage, formulation, sensor=sensor, entities=scenario,
                                                         sensor_budget=sensor_budget, use_sensor_cost=use_sensor_cost,
                                                         use_entity_weights=use_scenario_probability,
                                                         n_to_detect=n_to_detect,
                                                         coverage_col_name=coverage_col_name,
                                                         mip_solver_name=mip_solver_name,
                                                         pyomo_options=pyomo_options, solver_options=solver_options)


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

