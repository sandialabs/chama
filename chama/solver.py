"""
The solver module contains high-level solvers for sensor placement optimization.
"""
import pyomo.environ as pe
import chama.utils as cu
import numpy as np
import pandas as pd

class SensorPlacement:
    """
    This class implements a Pyomo-based sensor placement solver using the stochastic programming formulation
    from [LBSW12]_.

    Examples
    --------
    >>> # read the data into appropriate pandas DataFrame objects
    >>> ...
    >>> # create and call the solver
    >>> spsolver = SensorPlacement()
    >>> results = spsolver.solve(df_sensor, df_scenario, df_impact, 5)
    >>> # output the key results, e.g.,
    >>> print(results['selected_sensors'])
    """

    def __init__(self, **kwds):
        """
        Create an instance of the SensorPlacement
        """
        self.scenario_prob = kwds.pop('scenario_prob',False)
        
    def solve(self, df_sensor, df_scenario, df_impact, sensor_budget, mip_solver_name='glpk', pyomo_solver_options={}):
        """
        Call this method to solve the sensor placement problem using Pyomo.

        Parameters
        ----------
        df_sensor : pandas.DataFrame
            This is a pandas dataframe with columns "Sensor" (of type str) and "Cost" (of type float), where "Sensor"
            specifies the name of the sensor, and "Cost" gives the cost as a floating point number. For a simple sensor
            budget of N sensors, set the sensor_budget to N and specify the cost as 1.0 for each sensor.
        df_scenario : pandas.DataFrame
            This is a pandas dataframe with the columns "Scenario" (of type str) and "Undetected Impact" (of type float),
            where "Scenario" specifies the scenario name, and "Undetected Impact" specifies the impact that
            will be realized if this scenario is not detected by any selected sensor.
        df_impact : pandas.DataFrame
           This is a pandas dataframe with the columns "Scenario" (of type str), "Sensor" (of type str),
           and "Impact" (of type float). It is a sparse representation of an impact matrix where "Scenario" is
           the name of the scenario, "Sensor" is the name of the sensor, and "Impact" is the impact that will be
           realized if the particular scenario is detected FIRST by the particular sensor.
        sensor_budget : float
            The total budget available for purchase/installation of sensors. Solution will select a family of sensors
            whose combined cost is below the sensor_budget. For a simple sensor budget of N sensors, set this
            to N and the cost of each sensor to 1.0.
        mip_solver_name : str
            Specify the name of the optimization solver to be used. This string is passed to Pyomo,
            and the solver needs to be a solver that is supported by Pyomo. For the class of problem
            used here, the solver must support solution of mixed-integer programming problems.
        pyomo_solver_options : dict
            Specifies a dictionary of solver specific solver options to pass through the the Pyomo solver.

        Returns
        -------
            dict : a dictionary that includes the following fields.
                * selected_sensors : a list of the sensor names that were selected as part of the optimization
                * objective_value : the value of the objective at the optimal point (float)
                * scenario_detection : a dictionary with scenario name as keys and the *first* sensor to detect it as
                    the value. None indicates that the scenario was not detected.
        """

        # validate the pandas dataframe input
        cu.df_columns_required('df_sensor', df_sensor, {'Sensor': np.object, 'Cost': np.float64})
        cu.df_nans_not_allowed('df_sensor', df_sensor)
        cu.df_columns_required('df_scenario', df_scenario, {'Scenario': np.object, 'Undetected Impact': np.float64})
        cu.df_nans_not_allowed('df_scenario', df_scenario)
        cu.df_columns_required('df_impact', df_impact, {'Scenario': np.object, 'Sensor': np.object, 'Impact': np.float64})
        cu.df_nans_not_allowed('df_impact', df_impact)

        # validate optional columns in pandas dataframe input
        if self.scenario_prob:
            cu.df_columns_required('df_scenario',df_scenario, {'Probability':np.float64})

        model = self._create_pyomo_model(df_sensor, df_scenario, df_impact, sensor_budget)

        self._solve_pyomo_model(model, mip_solver_name, pyomo_solver_options)

        ret_dict = self._create_solution_summary(model)
        return ret_dict

    def _create_solution_summary(self, model):
        """
        Creates a dictionary representing common summary information about the solution from a Pyomo model object
        that has already been solved.

        Parameters
        ----------
        model : Pyomo model object
           This is the Pyomo model object that has already been solved.

        Returns
        -------
            dict : returns a dictionary object specified by :func:`~solver.SensorPlacement.solve`
        """
        selected_sensors = []
        for key in model.y:
            if pe.value(model.y[key]) > 0.5:
                selected_sensors.append(key)

        obj_value = pe.value(model.obj)

        scenario_detection = dict()
        for key in model.x:
            scenario = key[0]
            sensor = key[1]
            if pe.value(model.x[(scenario,sensor)]) > 0.5:
                scenario_detection[scenario] = sensor

        return {'selected_sensors': selected_sensors, 'objective_value': obj_value, 'scenario_detection': scenario_detection}
        
    def _create_pyomo_model(self, df_sensor, df_scenario, df_impact, sensor_budget):
        """
        Create and return the Pyomo model to be solved.

        Parameters
        ----------
        df_sensor : Pandas dataframe - see :func:`~solver.SensorPlacement.solve`
        df_scenario : Pandas dataframe - see :func:`~solver.SensorPlacement.solve`
        df_impact : Pandas dataframe - see :func:`~solver.SensorPlacement.solve`
        sensor_budget : float - see :func:`~solver.SensorPlacement.solve`

        Returns
        -------
            ConcreteModel : A Pyomo model ready to be solved
        """
        df_impact = df_impact.set_index(['Scenario','Sensor'])
        assert(df_impact.index.names[0] == 'Scenario')
        assert(df_impact.index.names[1] == 'Sensor')

        df_sensor = df_sensor.set_index('Sensor')
        assert(df_sensor.index.names[0] == 'Sensor')

        # Python set will extract the unique Scenario and Sensor values
        scenario_list = sorted(set(df_impact.index.get_level_values('Scenario')))
        sensor_list = sorted(set(df_impact.index.get_level_values('Sensor')))
        sensor_cost = df_sensor['Cost']

        # Add in the data for the dummy sensor to account for a scenario that is undetected
        sensor_list.append('__DUMMY_SENSOR_UNDETECTED__')

        df_dummy = pd.DataFrame(scenario_list, columns=['Scenario'])
        df_dummy = df_dummy.set_index(['Scenario'])

        df_scenario = df_scenario.set_index(['Scenario'])
        df_dummy['Impact'] = df_scenario['Undetected Impact']
        df_scenario.reset_index(level=0, inplace=True)

        df_dummy['Sensor'] = '__DUMMY_SENSOR_UNDETECTED__'
        df_dummy = df_dummy.reset_index().set_index(['Scenario', 'Sensor'])
        df_impact = df_impact.append(df_dummy)
        sensor_cost['__DUMMY_SENSOR_UNDETECTED__'] = 0

        # create a list of tuples for all the scenario/sensor pairs where
        # detection has occurred
        scenario_sensor_pairs = df_impact.index.tolist()

        # create the (jagged) index set of sensors that were able to detect a
        # particular scenario
        scenario_sensors = dict()
        for (a,i) in scenario_sensor_pairs:
            if a not in scenario_sensors:
                scenario_sensors[a] = list()
            scenario_sensors[a].append(i)

        # create the model container
        model = pe.ConcreteModel()
        model._scenario_sensors = scenario_sensors

        # Pyomo does not create an ordered dummy set when passed a list - do this for now as a workaround
        model.scenario_set = pe.Set(initialize=scenario_list, ordered=True)
        model.sensor_set = pe.Set(initialize=sensor_list, ordered=True)
        model.scenario_sensor_pairs_set = pe.Set(initialize=scenario_sensor_pairs, ordered=True)

        # x_{a,i} variable indicates which sensor is the first to detect scenario a
        model.x = pe.Var(model.scenario_sensor_pairs_set, bounds=(0,1))

        # y_i variable indicates if a sensor is installed or not
        model.y = pe.Var(model.sensor_set, within=pe.Binary)

        # objective function minimize the sum impact across all scenarios
        # in current formulation all scenarios are equally probable 
        def obj_rule(m):
            return 1.0/float(len(scenario_list))*sum(float(df_impact.loc[a,i])*m.x[a,i] for (a,i) in scenario_sensor_pairs)

        # Modify the objective function to include scenario probabilities
        if self.scenario_prob:
            df_scenario.set_index(['Scenario'], inplace=True)
            def obj_rule(m):
                return sum(float(df_scenario.loc[a,'Probability'])*float(df_impact.loc[a,i])*m.x[a,i] for (a,i) in scenario_sensor_pairs)

        model.obj = pe.Objective(rule=obj_rule)

        # constrain the problem to have only one x value for each scenario
        def limit_x_rule(m, a):
            return sum(m.x[a,i] for i in scenario_sensors[a]) == 1
        model.limit_x = pe.Constraint(model.scenario_set, rule=limit_x_rule)

        def detect_only_if_sensor_rule(m, a, i):
            return m.x[a,i] <= model.y[i]
        model.detect_only_if_sensor = pe.Constraint(model.scenario_sensor_pairs_set, rule=detect_only_if_sensor_rule)

        model.sensor_budget = pe.Constraint(expr=sum(sensor_cost[i]*model.y[i] for i in sensor_list) <= sensor_budget)

        return model

    def _solve_pyomo_model(self, model, mip_solver_name='glpk', pyomo_solver_options={}):
        """
        Solve the Pyomo model created to perform the sensor placement.

        Parameters
        ----------
        model : ConcreteModel
            A pyomo model representing the sensor placement problem
        mip_solver_name: str
            Name of the Pyomo solver to use when solving the problem
        pyomo_solver_options : dict
            A dictionary of solver specific options to pass through to the solver

        Returns
        -------
            Pyomo results object : returns a Pyomo solver results object
        """
        opt = pe.SolverFactory(mip_solver_name)
        return opt.solve(model, **pyomo_solver_options)

class SensorPlacement_ScenarioProbability(SensorPlacement):
    """
    A SensorPlacement that includes scenario probabilities in the formulation.
    """

    def __init__(self, *args, **kwds):
        kwds['scenario-prob'] = True
        SensorPlacement.__init__(self, *args, **kwds)
