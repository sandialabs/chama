from nose.tools import *
from os.path import abspath, dirname, join
import chama
import pandas as pd

testdir = dirname(abspath(__file__))
datadir = join(testdir,'data')

def test_net3_ec():
    # This test replicates WST sp_ex1
    impact_file = join(datadir, 'Net3_ec.impact')
    
    # read the impact file from the water simulations
    impact_data =  pd.read_csv(impact_file, skiprows=2, sep =' ', usecols=[0,1,3], names=['Scenario','Sensor','Impact'])
    impact_data['Scenario'] = impact_data['Scenario'].apply(str)    # convert the scenario names to strings
    impact_data['Sensor'] = impact_data['Sensor'].apply(str)        # convert the Sensor names to strings

    # Define sensor dataframe
    df_sensor = pd.DataFrame(data=impact_data['Sensor'].unique(), columns=['Sensor'])
    df_sensor = df_sensor[df_sensor.Sensor != "-1"]     # remove the "-1" sensors (used to indicate undetected impact)
    df_sensor['Cost'] = 1.0     # define the cost

    # Define scenario dataframe
    df_scenario = impact_data[impact_data.Sensor == "-1"]
    df_scenario = df_scenario.drop('Sensor', axis=1)
    df_scenario.rename(columns={'Impact': 'Undetected Impact'}, inplace=True)
    df_scenario.to_csv('blah_scenario.csv')

    # Define the impact data dataframe
    df_impact = impact_data[impact_data.Sensor != "-1"]

    # Solve sensor placement
    sensor_budget = 5
    solver = chama.solver.SPSensorPlacementSolver()
    results = solver.solve(df_sensor, df_scenario, df_impact, sensor_budget, pyomo_solver_options={'tee': True})

    expected_objective_value = 8655.80
    error = abs((results['objective_value'] - expected_objective_value)/expected_objective_value)
    assert_less(error, 0.01) # 1% error
    
    expected_selected_sensors = ["16", "21", "28", "38", "65", "__DUMMY_SENSOR_UNDETECTED__"]
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
    
if __name__ == '__main__':
    test_net3_ec()