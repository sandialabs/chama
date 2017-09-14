"""
The optimize module contains high-level solvers for sensor placement
optimization.
"""
from __future__ import print_function, division
import pyomo.environ as pe
import chama.utils as cu
import numpy as np
import pandas as pd

dummy_sensor_name = '__DUMMY_SENSOR_UNDETECTED__'


class Pmedian(object):
    """
    This class implements a Pyomo-based Pmedian sensor placement solver
    using the stochastic programming formulation from [LBSW12]_.
    """

    def __init__(self, **kwds):

        self.scenario_prob = kwds.pop('scenario_prob', False)
        self.coverage = kwds.pop('coverage', False)
        self._model = None
        self._sensor_df = None
        self._scenario_df = None
        self._impact_df = None
        
    def solve(self, sensor=None, scenario=None, impact=None,
              sensor_budget=None, mip_solver_name='glpk',
              pyomo_solver_options=None):
        """
        Call this method to solve the sensor placement problem using Pyomo.

        Parameters
        ----------
        sensor : :class:`pandas.DataFrame`
            This is a pandas DataFrame with columns "Sensor" (of type str) and
            "Cost" (of type float), where "Sensor" specifies the name of the
            sensor, and "Cost" gives the cost as a floating point number. For
            a simple sensor budget of N sensors, set the sensor_budget to N
            and specify the cost as 1.0 for each sensor.
        scenario : :class:`pandas.DataFrame`
            This is a pandas DataFrame with the columns "Scenario" (of type
            str) and "Undetected Impact" (of type float), where "Scenario"
            specifies the scenario name, and "Undetected Impact" specifies the
            impact that will be realized if this scenario is not detected by
            any selected sensor. This DataFrame may also include the
            optional column "Probability" (of type float) to specify the
            probability of each scenario. Scenario probability values will
            only be used if the ``scenario_prob`` flag is set to True.
        impact : :class:`pandas.DataFrame`
            This is a pandas DataFrame with the columns "Scenario" (of type
            str), "Sensor" (of type str), and "Impact" (of type float). It is
            a sparse representation of an impact matrix where "Scenario" is
            the name of the scenario, "Sensor" is the name of the sensor, and
            "Impact" is the impact that will be realized if the particular
            scenario is detected FIRST by the particular sensor.
        sensor_budget : float
            The total budget available for purchase/installation of sensors.
            Solution will select a family of sensors whose combined cost is
            below the sensor_budget. For a simple sensor budget of N sensors,
            set this to N and the cost of each sensor to 1.0.
        mip_solver_name : str
            Specify the name of the optimization solver to be used. This string
            is passed to Pyomo, and the solver needs to be a solver that is
            supported by Pyomo. For the class of problem used here, the solver
            must support solution of mixed-integer programming problems.
        pyomo_solver_options : dict
            Specifies a dictionary of solver specific solver options to pass
            through the the Pyomo solver. Defaults to an empty dictionary

        Returns
        -------
        ret_dict : dict
            dictionary that includes the following fields:
                * selected_sensors : a list of the sensor names that were
                  selected as part of the optimization
                * objective_value : the value of the objective at the optimal
                  point (float)
                * scenario_detection : a dictionary with scenario name as keys
                  and the **first** sensor to detect it as the value. None
                  indicates that the scenario was not detected.

        """

        if pyomo_solver_options is None:
            pyomo_solver_options = {}
        
        if self._model is None:
            model = self.create_pyomo_model(sensor, scenario, impact,
                                            sensor_budget)
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
        dict
            Dictionary containing objective value, selected sensors, and 
            impact assesment.
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
        
        return {'Objective': obj_value,
                'Sensors': selected_sensors,
                'Assessment': selected_impact}

    def create_pyomo_model(self, df_sensor, df_scenario, df_impact,
                            sensor_budget):
        """
        Create and return the Pyomo model to be solved.

        Parameters
        ----------
        df_sensor : :class:`pandas.DataFrame`
            see :func:`~solver.Pmedian.solve`
        df_scenario : :class:`pandas.DataFrame`
            see :func:`~solver.Pmedian.solve`
        df_impact : :class:`pandas.DataFrame`
            see :func:`~solver.Pmedian.solve`
        sensor_budget : float
            see :func:`~solver.Pmedian.solve`

        Returns
        -------
        ConcreteModel
            A Pyomo model ready to be solved
        """

        # validate the pandas dataframe input
        cu.df_columns_required('df_sensor', df_sensor,
                               {'Sensor': np.object,
                                'Cost': [np.float64, np.int64]})
        cu.df_nans_not_allowed('df_sensor', df_sensor)
        cu.df_columns_required('df_scenario', df_scenario,
                               {'Scenario': np.object,
                                'Undetected Impact': [np.float64, np.int64]})
        cu.df_nans_not_allowed('df_scenario', df_scenario)
        cu.df_columns_required('df_impact', df_impact,
                               {'Scenario': np.object,
                                'Sensor': np.object,
                                'Impact': [np.float64, np.int64]})
        cu.df_nans_not_allowed('df_impact', df_impact)

        # validate optional columns in pandas dataframe input
        if self.scenario_prob:
            cu.df_columns_required('df_scenario', df_scenario,
                                   {'Probability': np.float64})

        self._sensor_df = df_sensor
        self._scenario_df = df_scenario
        self._impact_df = df_impact

        df_impact = df_impact.set_index(['Scenario', 'Sensor'])
        assert(df_impact.index.names[0] == 'Scenario')
        assert(df_impact.index.names[1] == 'Sensor')

        df_sensor = df_sensor.set_index('Sensor')
        assert(df_sensor.index.names[0] == 'Sensor')

        # Python set will extract the unique Scenario and Sensor values
        scenario_list = \
            sorted(set(df_impact.index.get_level_values('Scenario')))
        sensor_list = sorted(set(df_impact.index.get_level_values('Sensor')))
        sensor_cost = df_sensor['Cost']

        # Add in the data for the dummy sensor to account for a scenario that
        # is undetected
        sensor_list.append(dummy_sensor_name)

        df_dummy = pd.DataFrame(scenario_list, columns=['Scenario'])
        df_dummy = df_dummy.set_index(['Scenario'])

        df_scenario = df_scenario.set_index(['Scenario'])
        df_dummy['Impact'] = df_scenario['Undetected Impact']
        df_scenario.reset_index(level=0, inplace=True)

        df_dummy['Sensor'] = dummy_sensor_name
        df_dummy = df_dummy.reset_index().set_index(['Scenario', 'Sensor'])
        df_impact = df_impact.append(df_dummy)
        sensor_cost[dummy_sensor_name] = 0.0

        # create a list of tuples for all the scenario/sensor pairs where
        # detection has occurred
        scenario_sensor_pairs = df_impact.index.tolist()

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
                   sum(float(df_impact.loc[a, i]) * m.x[a, i]
                       for (a, i) in scenario_sensor_pairs)

        # Modify the objective function to include scenario probabilities
        if self.scenario_prob:
            df_scenario.set_index(['Scenario'], inplace=True)

            def obj_rule(m):
                return sum(float(df_scenario.loc[a, 'Probability']) *
                           float(df_impact.loc[a, i]) * m.x[a, i]
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
        Solve the Pyomo model created to perform the sensor placement.

        Parameters
        ----------
        model : ConcreteModel
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
        Add a sensor grouping constraint to the sensor placement model. This
        constraint forces a certain number of sensors to be selected from a
        particular subset of all the possible sensors

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


class Pmedian_ScenarioProbability(Pmedian):
    """
    A Pmedian that includes scenario probabilities in the formulation.
    """

    def __init__(self, **kwds):
        kwds['scenario-prob'] = True
        Pmedian.__init__(self, **kwds)


class Coverage(Pmedian):
    
    def __init__(self, **kwds):
        kwds['coverage'] = True
        Pmedian.__init__(self, **kwds)

    def create_pyomo_model(self, df_sensor, df_scenario, df_impact,
                            sensor_budget):
        """
        Modify the input DataFrames to be compatible with the coverage
        formulation. Call the super class implementation to actually build
        the Pyomo model.
        """

        df_impact, df_scenario = self._detection_times_to_coverage(df_impact)

        model = Pmedian.create_pyomo_model(self, df_sensor, df_scenario,
                                           df_impact, sensor_budget)
        return model

    def _detection_times_to_coverage(self, det_times):
    
        temp = {'Scenario': [], 'Sensor': [], 'Impact': []}
        for index, row in det_times.iterrows():
            for t in row['Impact']:
                temp['Scenario'].append(str((t, row['Scenario'])))
                temp['Sensor'].append(row['Sensor'])
                temp['Impact'].append(0.0)

        coverage = pd.DataFrame()
        coverage['Scenario'] = temp['Scenario']
        coverage['Sensor'] = temp['Sensor']
        coverage['Impact'] = temp['Impact']
        coverage = coverage.sort_values('Scenario')
        coverage = coverage.reset_index(drop=True)

        scenarios = pd.DataFrame()
        scenarios['Scenario'] = coverage['Scenario'].unique()
        scenarios['Undetected Impact'] = 1.0
        
        return coverage, scenarios
