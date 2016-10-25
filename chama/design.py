"""
Pyomo models for sensor design optimization
"""
import pyomo.environ as pe
from pyomo.opt import SolverFactory

class PyomoDesignSolver:
    """
    This class is responsible for solving sensor design problems using Pyomo. Currently, only one
    problem formulation is supported (standard p-median).
    """
    
    def __init__(self, solver):
        """
        Create an instance of the PyomoDesignSolver

        Parameters
        ----------
        solver : str
            Specify the name of the optimization solver to be used. This string is passed to Pyomo,
            and the solver needs to be a solver that is supported by Pyomo. For the class of problem
            used here, the solver must support solution of mixed-integer programming problems
        """
        self._solver = solver
    
    def solve(self, impact, sensor_dict, sensor_budget, tee=False, keepfiles=False):
        """
        Call this method to solve the sensor placement problem using Pyomo.

        Parameters
        ----------
        impact : pandas ?
            Specifies... ?
        sensor_dict : dict
            The dictionary of all sensor objects that can be selected. Keys are the sensor "names" and must be unique
        sensor_budget : float
            The total budget available for purchase/installation of sensors. Solution will select a family of sensors
            whose combined cost is below the sensor_budget

        Returns
        -------
            dict : a dictionary that includes two fields.
                * selected_sensors : a list of the sensor names that were selected as part of the optimization
                * objective_value : the value of the objective at the optimal point (float)
        """
        model = self._create_pyomo_model(impact, sensor_dict, sensor_budget)
        self._solve_pyomo_model(model, tee=tee, keepfiles=keepfiles)

        selected_sensors = []
        sensor_impact = dict()
        for key in model.y:
            if pe.value(model.y[key]) > 0.5:
                sensor_impact[key] = 0
                selected_sensors.append(key)

        obj_value = pe.value(model.obj)

        for key in model.x:
            if pe.value(model.x[key]) > 0.5:
                sensor_impact[key[1]] += 1
                
        return {'selected_sensors': selected_sensors, 'objective_value': obj_value, 'sensor_impact': sensor_impact}
        
    def _create_pyomo_model(self, impact, sensor_dict, sensor_budget):
        assert(impact.index.names[0] == 'Scenario')
        assert(impact.index.names[1] == 'Sensor')

        scenario_set = set(impact.index.get_level_values('Scenario'))
        sensor_set = set(impact.index.get_level_values('Sensor'))

        # create a set of tuples for all the scenario/sensor pairs where
        # detection has occurred
        # bug in pyomo, this has to be a list, and cannot be a set.
        scenario_sensor_pairs = impact.index.tolist()

        # create the (jagged) index set of sensors that were able to detect a
        # particular scenario
        scenario_sensors = dict()
        for (a,i) in scenario_sensor_pairs:
            if a not in scenario_sensors:
                scenario_sensors[a] = set()
            scenario_sensors[a].add(i)

        # create the model container
        model = pe.ConcreteModel()

        # x_{a,i} variable indicates which sensor is the first to detect scenario a
        model.x = pe.Var(scenario_sensor_pairs, bounds=(0,1))

        # y_i variable indicates if a sensor is installed or not
        model.y = pe.Var(sensor_set, within=pe.Binary)

        # objective function minimize the sum impact across all scenarios
        def obj_rule(m):
            return 1.0/float(len(scenario_set))*sum(float(impact.loc[a,i])*m.x[a,i] for (a,i) in scenario_sensor_pairs)
        model.obj = pe.Objective(rule=obj_rule)

        # constrain the problem to have only one x value for each scenario
        def limit_x_rule(m, a):
            return sum(m.x[a,i] for i in scenario_sensors[a]) == 1
        model.limit_x = pe.Constraint(scenario_set, rule=limit_x_rule)

        def detect_only_if_sensor_rule(m, a, i):
            return m.x[a,i] <= model.y[i]
        model.detect_only_if_sensor = pe.Constraint(scenario_sensor_pairs, rule=detect_only_if_sensor_rule)

        model.sensor_budget = pe.Constraint(expr=sum(sensor_dict[i].cost*model.y[i] for i in sensor_set if i != -1 and i != '_NotDetected') <= sensor_budget)

        return model

    def _solve_pyomo_model(self, model, tee=False, keepfiles=False):
        """
        Solve a pyomo model
        """
        opt = SolverFactory(self._solver)
        opt.solve(model, tee=tee, keepfiles=keepfiles)
