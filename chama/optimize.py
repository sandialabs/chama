"""
The optimize module contains high-level solvers for sensor placement
optimization.
"""
from __future__ import print_function, division
import pyomo.environ as pe
import chama.utils as cu
import numpy as np
import pandas as pd
from pyomo.opt import SolverStatus, TerminationCondition

dummy_sensor_name = '__DUMMY_SENSOR_UNDETECTED__'


class Pmedian(object):
    """
    Pyomo-based Pmedian sensor placement optimization.
    """

    def __init__(self):
        self._model = None
        self._impact = None
        self._sensor = None
        self._scenario = None
        self._use_scenario_probability = False
        self._use_sensor_cost = False
        self._solved = False

    def solve(self, impact=None, sensor=None, scenario=None,
              sensor_budget=None, use_sensor_cost=False,
              use_scenario_probability=False, impact_col_name='Impact',
              mip_solver_name='glpk', pyomo_solver_options=None):
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
        pyomo_solver_options : dict
            Solver specific options to pass through Pyomo.
            Defaults to an empty dictionary.

        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean impact based on the selected sensors
            * Assessment: The impact value for each sensor-scenario pair.
              The assessment is stored as a pandas DataFrame with columns
              'Scenario', 'Sensor', and 'Impact' (same format as the input
              Impact assessment) If the selected sensors did not detect a
              particular scenario, the impact is set to the Undetected Impact.
        """

        self.create_pyomo_model(impact, sensor, scenario, sensor_budget,
                                use_sensor_cost, use_scenario_probability,
                                impact_col_name)

        self.solve_pyomo_model(sensor_budget, mip_solver_name,
                               pyomo_solver_options)

        results_dict = self.create_solution_summary()

        return results_dict

    def create_pyomo_model(self, impact, sensor, scenario, sensor_budget,
                           use_sensor_cost=False,
                           use_scenario_probability=False,
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
        # reset the internal model and data attributes
        # BLN: Why do we reset these when they will be overwritten at the
        # end of this method anyway?
        self._model = None
        self._impact = None
        self._sensor = None
        self._scenario = None

        # validate the pandas dataframe input
        cu._df_columns_required('sensor', sensor,
                                {'Sensor': np.object})
        cu._df_nans_not_allowed('sensor', sensor)
        cu._df_columns_required('scenario', scenario,
                                {'Scenario': np.object,
                                 'Undetected Impact': [np.float64, np.int64]})
        cu._df_nans_not_allowed('scenario', scenario)
        cu._df_columns_required('impact', impact,
                                {'Scenario': np.object,
                                 'Sensor': np.object,
                                 impact_col_name: [np.float64, np.int64]})
        cu._df_nans_not_allowed('impact', impact)

        # validate optional columns in pandas dataframe input
        if use_scenario_probability:
            cu._df_columns_required('scenario', scenario,
                                    {'Probability': np.float64})

        if use_sensor_cost:
            cu._df_columns_required('sensor', sensor,
                                    {'Cost': [np.float64, np.int64]})

        # Notice, setting the index here
        impact = impact.set_index(['Scenario', 'Sensor'])
        assert(impact.index.names[0] == 'Scenario')
        assert(impact.index.names[1] == 'Sensor')

        # Notice, setting the index here
        sensor = sensor.set_index('Sensor')
        assert(sensor.index.names[0] == 'Sensor')

        # Python set will extract the unique Scenario and Sensor values
        scenario_list = \
            sorted(set(impact.index.get_level_values('Scenario')))
        sensor_list = sorted(set(impact.index.get_level_values('Sensor')))
        if use_sensor_cost:
            sensor_cost = sensor['Cost']
        else:
            sensor['Cost'] = 1
            sensor_cost = sensor['Cost']

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
        model._scenario_sensors = scenario_sensors

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
        model.sensor_budget_con = \
            pe.Constraint(expr=sum(float(sensor_cost[i]) * model.y[i]
                                   for i in sensor_list) <= model.sensor_budget)

        self._model = model
        impact.reset_index(inplace=True)
        self._impact = impact
        self._sensor = sensor
        scenario.reset_index(inplace=True)
        self._scenario = scenario
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

        # TODO: Should we make this easier by just allowing lower bound and
        upper bound and do an equality if they are the same?

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
            if min_select < 0 or max_select:
                raise ValueError('Cannot select a negative number of sensors')

            if min_select > max_select:
                raise ValueError('min_select must be less than max_select')

            gconlist.add(min_select <= sensor_sum <= max_select)

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

    def solve_pyomo_model(self, sensor_budget=None, mip_solver_name='glpk',
                          pyomo_solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        See :py:meth:`Pmedian.solve` for more information on arguments.

        Parameters
        ----------
        sensor_budget : float
            Sensor budget
        mip_solver_name: str
            Name of the Pyomo solver to use when solving the problem
        pyomo_solver_options : dict
            A dictionary of solver specific options to pass through to the
            solver. Defaults to an empty dictionary

        Returns
        -------
        Pyomo results object
        """
        if pyomo_solver_options is None:
            pyomo_solver_options = {}

        if self._model is None:
            raise RuntimeError('Cannot call solve_pyomo_model before the model'
                               ' is created with create_pyomo_model'
                               )

        # change the sensor budget if necessary
        if sensor_budget is not None:
            self._model.sensor_budget = sensor_budget

        # create the solver
        opt = pe.SolverFactory(mip_solver_name)

        # TODO: Update description of pyomo_solver_options because technically
        # solver-specific options would have to be passed as {'tee':True,
        # 'options':{dict with solver options}}
        # solve the problem and return the result
        results = opt.solve(self._model, **pyomo_solver_options)

        # Check solver status
        if (results.solver.status == SolverStatus.ok) and \
           (results.solver.termination_condition == TerminationCondition.optimal):
            self._solved = True
        else:
            self._solved = False
            print('The solver was unable to find an optimal solution')

        return results

    def create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        Returns
        -------
        Dictionary containing objective value, selected sensors, and
        impact assessment.
        """

        if self._model is None or not self._solved:
            raise RuntimeError('Cannot call create_solution_summary before '
                               'the model is created and solved.'
                               )

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

        if type(self) is Coverage:
            obj_value = 1 - obj_value
            selected_impact['Impact'] = 1 - selected_impact['Impact']

        return {'Objective': obj_value,
                'Sensors': selected_sensors,
                'Assessment': selected_impact}

class OldPmedian(object):
    """
    Pyomo-based Pmedian sensor placement optimization.
    """

    def __init__(self, use_sensor_cost=False, use_scenario_probability=False):
        """
        Parameters
        ----------
        use_sensor_cost : bool
            Boolean indicating if sensor cost should be used in the optimization.
            If False, sensors have equal cost of 1.
        use_scenario_probability : bool
            Boolean indicating if scenario probability should be used in the optimization.
            If False, scenarios have equal probability.
        """
        
        self.use_sensor_cost = use_sensor_cost
        self.use_scenario_probability = use_scenario_probability
        self._model = None
        self._sensor_df = None
        self._scenario_df = None
        self._impact_df = None
        
    def solve(self, impact=None, sensor_budget=None, sensor=None, scenario=None, 
              mip_solver_name='glpk', pyomo_solver_options=None):
        """
        Solves the sensor placement optimization.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact assessment. A single detection time (or other measure 
            of damage) for each sensor that detects a scenario. 
            Impact is stored as a pandas DataFrmae with columns 'Scenario', 
            'Sensor', 'Impact'.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the 'use_sensor_cost' to False.
        sensor : pandas DataFrame
            Sensor characteristics.  Contains sensor cost for each sensor. 
            Sensor characteristics are stored as a pandas DataFrame with 
            columns 'Sensor' and 'Cost'. Cost is used in the sensor 
            placement optimization if the 'use_sensor_cost' flag is set to True.
        scenario : pandas DataFrame
            Scenario characteristics.  Contains scenario probability and the 
            impact for undetected scenarios. Scenario characteristics are 
            stored as a pandas DataFrame with columns 'Scenario', 
            'Undetected Impact', and 'Probability'. Undetected Impact is 
            required for each scenario. Probability is used if the 
            'use_scenario_probability' flag is set to True.
        mip_solver_name : str
            Optimization solver name passed to Pyomo. The solver must be 
            supported by Pyomo and support solution of mixed-integer 
            programming problems.
        pyomo_solver_options : dict
            Solver specific options to pass through Pyomo. 
            Defaults to an empty dictionary.

        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean impact based on the selected sensors
            * Assessment: The impact value for each sensor-scenario pair. 
              The assessment is stored as a pandas DataFrame with columns 
              'Scenario', 'Sensor', and 'Impact' (same format as the input 
              Impact assessment) If the selected sensors did not detect a 
              particular scenario, the impact is set to the Undetected Impact.
        """

        if pyomo_solver_options is None:
            pyomo_solver_options = {}
        
        #if type(self) is Coverage:
        #    impact, scenario = self._detection_times_to_coverage(impact, scenario)
            
        if self._model is None:
            model = self.create_pyomo_model(impact, sensor_budget, sensor, 
                                            scenario)
        else:
            model = self._model

        self._solve_pyomo_model(model, mip_solver_name, pyomo_solver_options)

        ret_dict = self._create_solution_summary()
        return ret_dict

    def _create_solution_summary(self):
        """
        Creates a dictionary representing common summary information about the
        solution from a Pyomo model object that has already been solved.

        Returns
        -------
        Dictionary containing objective value, selected sensors, and 
        impact assessment.
        """
        model = self._model
        impact_df = self._impact_df
        scenario_df = self._scenario_df

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
        
        if type(self) is Coverage:
            obj_value = 1 - obj_value
            selected_impact['Impact'] = 1 - selected_impact['Impact']
            
        return {'Objective': obj_value,
                'Sensors': selected_sensors,
                'Assessment': selected_impact}

    def create_pyomo_model(self, impact, sensor_budget, sensor, scenario):
        """
        Returns the Pyomo model.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact assessment
        sensor_budget : float
            Sensor budget
        sensor : pandas DataFrame
            Sensor characteristics
        scenario : pandas DataFrame
            Scenario characteristics
            
        Returns
        -------
        Pyomo ConcreteModel ready to be solved
        """

        # validate the pandas dataframe input
        cu._df_columns_required('sensor', sensor,
                               {'Sensor': np.object,
                                'Cost': [np.float64, np.int64]})
        cu._df_nans_not_allowed('sensor', sensor)
        cu._df_columns_required('scenario', scenario,
                               {'Scenario': np.object,
                                'Undetected Impact': [np.float64, np.int64]})
        cu._df_nans_not_allowed('scenario', scenario)
        cu._df_columns_required('impact', impact,
                               {'Scenario': np.object,
                                'Sensor': np.object,
                                'Impact': [np.float64, np.int64]})
        cu._df_nans_not_allowed('impact', impact)

        # validate optional columns in pandas dataframe input
        if self.use_scenario_probability:
            cu._df_columns_required('scenario', scenario,
                                   {'Probability': np.float64})

        self._sensor_df = sensor
        self._scenario_df = scenario
        self._impact_df = impact

        impact = impact.set_index(['Scenario', 'Sensor'])
        assert(impact.index.names[0] == 'Scenario')
        assert(impact.index.names[1] == 'Sensor')

        sensor = sensor.set_index('Sensor')
        assert(sensor.index.names[0] == 'Sensor')

        # Python set will extract the unique Scenario and Sensor values
        scenario_list = \
            sorted(set(impact.index.get_level_values('Scenario')))
        sensor_list = sorted(set(impact.index.get_level_values('Sensor')))
        if self.use_sensor_cost:
            sensor_cost = sensor['Cost']
        else:
            sensor['Cost'] = 1
            sensor_cost = sensor['Cost']

        # Add in the data for the dummy sensor to account for a scenario that
        # is undetected
        sensor_list.append(dummy_sensor_name)

        df_dummy = pd.DataFrame(scenario_list, columns=['Scenario'])
        df_dummy = df_dummy.set_index(['Scenario'])

        scenario = scenario.set_index(['Scenario'])
        df_dummy['Impact'] = scenario['Undetected Impact']
        scenario.reset_index(level=[0], inplace=True)

        df_dummy['Sensor'] = dummy_sensor_name
        df_dummy = df_dummy.reset_index().set_index(['Scenario', 'Sensor'])
        impact = impact.append(df_dummy)
        sensor_cost[dummy_sensor_name] = 0.0

        # create a list of tuples for all the scenario/sensor pairs where
        # detection has occurred
        scenario_sensor_pairs = impact.index.tolist()

        # create the (jagged) index set of sensors that were able to detect a
        # particular scenario
        scenario_sensors = dict()
        for (a, i) in scenario_sensor_pairs:
            if a not in scenario_sensors:
                scenario_sensors[a] = list()
            scenario_sensors[a].append(i)

        # create the model container
        model = pe.ConcreteModel()
        model._scenario_sensors = scenario_sensors

        # Pyomo does not create an ordered dummy set when passed a list - do
        # this for now as a workaround
        model.scenario_set = pe.Set(initialize=scenario_list, ordered=True)
        model.sensor_set = pe.Set(initialize=sensor_list, ordered=True)
        model.scenario_sensor_pairs_set = \
            pe.Set(initialize=scenario_sensor_pairs, ordered=True)

        # x_{a,i} variable indicates which sensor is the first to detect
        # scenario a
        model.x = pe.Var(model.scenario_sensor_pairs_set, bounds=(0, 1))

        # y_i variable indicates if a sensor is installed or not
        model.y = pe.Var(model.sensor_set, within=pe.Binary)

        # objective function minimize the sum impact across all scenarios
        # in current formulation all scenarios are equally probable 
        def obj_rule(m):
            return 1.0 / float(len(scenario_list)) * \
                   sum(float(impact.at[a, i]) * m.x[a, i]
                       for (a, i) in scenario_sensor_pairs)

        # Modify the objective function to include scenario probabilities
        if self.use_scenario_probability:
            scenario.set_index(['Scenario'], inplace=True)

            def obj_rule(m):
                return sum(float(scenario.at[a, 'Probability']) *
                           float(impact.at[a, i]) * m.x[a, i]
                           for (a, i) in scenario_sensor_pairs)

        model.obj = pe.Objective(rule=obj_rule)

        # constrain the problem to have only one x value for each scenario
        def limit_x_rule(m, a):
            return sum(m.x[a, i] for i in scenario_sensors[a]) == 1
        model.limit_x = pe.Constraint(model.scenario_set, rule=limit_x_rule)

        def detect_only_if_sensor_rule(m, a, i):
            return m.x[a, i] <= model.y[i]
        model.detect_only_if_sensor = \
            pe.Constraint(model.scenario_sensor_pairs_set,
                          rule=detect_only_if_sensor_rule)

        model.sensor_budget = \
            pe.Constraint(expr=sum(float(sensor_cost[i]) * model.y[i]
                                   for i in sensor_list) <= sensor_budget)

        self._model = model

        return model

    def _solve_pyomo_model(self, model, mip_solver_name='glpk',
                           pyomo_solver_options=None):
        """
        Solves the Pyomo model created to perform the sensor placement.

        Parameters
        ----------
        model : pyomo ConcreteModel
            A pyomo model representing the sensor placement problem
        mip_solver_name: str
            Name of the Pyomo solver to use when solving the problem
        pyomo_solver_options : dict
            A dictionary of solver specific options to pass through to the
            solver. Defaults to an empty dictionary

        Returns
        -------
        Pyomo results object
        """
        if pyomo_solver_options is None:
            pyomo_solver_options = {}

        opt = pe.SolverFactory(mip_solver_name)
        return opt.solve(model, **pyomo_solver_options)

    def add_grouping_constraint(self, sensor_list, select=None,
                                min_select=None, max_select=None):
        """
        Adds a sensor grouping constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors.

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
            self._model.add_component('_groupingconlist', pe.ConstraintList())
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
            if min_select < 0 or max_select:
                raise ValueError('Cannot select a negative number of sensors')

            if min_select > max_select:
                raise ValueError('min_select must be less than max_select')

            gconlist.add(min_select <= sensor_sum <= max_select)

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

class Coverage(object):
    """
    A translation of the Pyomo-based Pmedian sensor placement optimization used 
    to optimize coverage.
    """
    
    def __init__(self):
        self._pmedian_solver = Pmedian()
        self._coverage_type = None

    def solve(self, impact=None, sensor=None, scenario=None, sensor_budget=None,
              use_sensor_cost=False, use_scenario_probability=False, coverage_type='scenario',
              impact_col_name='Impact', mip_solver_name='glpk', pyomo_solver_options=None):

        """
        Solves the sensor placement optimization using Coverage.

        Parameters
        ----------
        impact : pandas DataFrame
            Impact assessment. A single detection time (or other measure
            of damage) for each sensor that detects a scenario.
            Impact is stored as a pandas DataFrmae with columns 'Scenario',
            'Sensor', 'Impact'.
        sensor : pandas DataFrame
            Sensor characteristics.  Contains sensor cost for each sensor.
            Sensor characteristics are stored as a pandas DataFrame with
            columns 'Sensor' and 'Cost'. Cost is used in the sensor
            placement optimization if the 'use_sensor_cost' flag is set to True.
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
            Boolean indicating if sensor cost should be used in the optimization.
            If False, sensors have equal cost of 1.
        use_scenario_probability : bool
            Boolean indicating if scenario probability should be used in the optimization.
            If False, scenarios have equal probability.
        coverage_type : 'scenario' or 'scenario-time'
            String indicating if coverage is determined on a scenario basis or scenario-time (including each time
            as a separate scenario)
        impact_col_name : str
            The name of the column containing the impact data to be used
            in the objective function.
        mip_solver_name : str
            Optimization solver name passed to Pyomo. The solver must be
            supported by Pyomo and support solution of mixed-integer
            programming problems.
        pyomo_solver_options : dict
            Solver specific options to pass through Pyomo.
            Defaults to an empty dictionary.

        Returns
        -------
        A dictionary with the following keys:
            * Sensors: A list of the selected sensors
            * Objective: The mean impact based on the selected sensors
            * Assessment: The impact value for each sensor-scenario pair. 
              The assessment is stored as a pandas DataFrame with columns 
              'Scenario', 'Sensor', and 'Impact' (same format as the input 
              Impact assessment) If the selected sensors did not detect a 
              particular scenario, the impact is set to the Undetected Impact.
        """

        impact, scenario = self.convert_detection_times_to_coverage(impact, scenario)

        self.create_pyomo_model(impact, sensor, scenario, sensor_budget, use_sensor_cost, use_scenario_probability, impact_col_name)

        self.solve_pyomo_model(sensor_budget, mip_solver_name, pyomo_solver_options)

        results_dict = self.create_solution_summary()

        return results_dict


    def convert_detection_times_to_coverage(self, det_times, scenario, use_scenario_probability=False, coverage_type='scenario'):
        
        temp = {'Scenario': [], 'Sensor': [], 'Impact': []}
        prob_time = {}
        for index, row in det_times.iterrows():
            if coverage_type == 'scenario':
                temp['Scenario'].append(row['Scenario'])
                temp['Sensor'].append(row['Sensor'])
                temp['Impact'].append(0.0)
            elif coverage_type=='scenario-time':
                for t in row['Impact']:
                    scen_name = str((t, row['Scenario']))
                    temp['Scenario'].append(scen_name)
                    temp['Sensor'].append(row['Sensor'])
                    temp['Impact'].append(0.0)
                    if use_scenario_probability:
                        prob_time[scen_name] = scenario[scenario['Scenario'] == row['Scenario']]['Probability'].values[0]
                    
        coverage = pd.DataFrame()
        coverage['Scenario'] = temp['Scenario']
        coverage['Sensor'] = temp['Sensor']
        coverage['Impact'] = temp['Impact']
        coverage = coverage.sort_values('Scenario')
        coverage = coverage.reset_index(drop=True)

        updated_scenario = pd.DataFrame()  
        if coverage_type=='scenario':
            updated_scenario['Scenario'] = scenario['Scenario']
            updated_scenario['Undetected Impact'] = 1.0
            if use_scenario_probability:
                updated_scenario['Probability'] = scenario['Probability']
        else:          
            updated_scenario['Scenario'] = coverage['Scenario'].unique()
            updated_scenario['Undetected Impact'] = 1.0
            if use_scenario_probability:
                updated_scenario.set_index('Scenario', inplace=True)
                updated_scenario['Probability'] = pd.Series(prob_time)
                updated_scenario.reset_index(inplace=True)
        
        return coverage, updated_scenario
