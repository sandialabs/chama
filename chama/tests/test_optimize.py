from nose.tools import *
from nose.plugins.skip import SkipTest
from os.path import abspath, dirname, join
import pandas as pd
import numpy as np
import chama

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

def test_water_network_example():
    # This test replicates WST sp_ex1
    impact_file = join(datadir, 'Net3_ec.impact')
    
    # read the impact file from the water simulations
    impact_data = pd.read_csv(impact_file, skiprows=2, sep=' ',
                               usecols=[0, 1, 3],
                               names=['Scenario', 'Sensor', 'Impact'])

    # convert the scenario names to strings
    impact_data['Scenario'] = impact_data['Scenario'].apply(str)
    # convert the Sensor names to strings
    impact_data['Sensor'] = impact_data['Sensor'].apply(str)

    # Define sensor dataframe
    df_sensor = pd.DataFrame(data=impact_data['Sensor'].unique(),
                             columns=['Sensor'])
    # remove the "-1" sensors (used to indicate undetected impact)
    df_sensor = df_sensor[df_sensor.Sensor != "-1"]
    df_sensor['Cost'] = 1.0     # define the cost

    # Define scenario dataframe
    df_scenario = impact_data[impact_data.Sensor == "-1"]
    df_scenario = df_scenario.drop('Sensor', axis=1)
    df_scenario.rename(columns={'Impact': 'Undetected Impact'}, inplace=True)
    # df_scenario.to_csv('blah_scenario.csv')

    # Define the impact data dataframe
    df_impact = impact_data[impact_data.Sensor != "-1"]

    # Solve sensor placement
    sensor_budget = 5
    solver = chama.optimize.Pmedian()
    results = solver.solve(df_sensor, df_scenario, df_impact, sensor_budget,
                           pyomo_solver_options={'tee': False})

    expected_objective_value = 8655.80
    error = abs((results['objective_value'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    
    expected_selected_sensors = ["16", "21", "28", "38", "65",
                                 "__DUMMY_SENSOR_UNDETECTED__"]
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
    

def test_water_network_example_with_scenario_prob():
    # This test replicates WST sp_ex1
    impact_file = join(datadir, 'Net3_ec.impact')
    
    # read the impact file from the water simulations
    impact_data = pd.read_csv(impact_file, skiprows=2, sep=' ',
                               usecols=[0, 1, 3],
                               names=['Scenario', 'Sensor', 'Impact'])

    # convert the scenario names to strings
    impact_data['Scenario'] = impact_data['Scenario'].apply(str)
    # convert the Sensor names to strings
    impact_data['Sensor'] = impact_data['Sensor'].apply(str)

    # Define sensor dataframe
    df_sensor = pd.DataFrame(data=impact_data['Sensor'].unique(),
                             columns=['Sensor'])

    # remove the "-1" sensors (used to indicate undetected impact)
    df_sensor = df_sensor[df_sensor.Sensor != "-1"]
    df_sensor['Cost'] = 1.0  # define the cost

    # Define scenario dataframe
    df_scenario = impact_data[impact_data.Sensor == "-1"]
    df_scenario = df_scenario.drop('Sensor', axis=1)
    df_scenario.rename(columns={'Impact': 'Undetected Impact'}, inplace=True)
    # df_scenario.to_csv('blah_scenario.csv')

    # Define the impact data dataframe
    df_impact = impact_data[impact_data.Sensor != "-1"]

    # Add scenario probabilities
    df_scenario['Probability'] = 0.0041
    df_scenario.set_index('Scenario', inplace=True)
    df_scenario.loc['165', 'Probability'] = \
        1.0 - sum(df_scenario.iloc[1:].Probability)
    # Changing the undetected impact of scenario 165 such that the scenario
    # is not detected when scenario probabilities are ignored but forces a
    # different selection of sensors when scenario probabilities are
    # incorporated. 
    df_scenario.loc['165', 'Undetected Impact'] = 30000.0
    df_scenario.reset_index(inplace=True)

    # Solve sensor placement
    sensor_budget = 5
    use_prob = False
    solver = chama.optimize.Pmedian(scenario_prob=use_prob)
    results = solver.solve(df_sensor, df_scenario, df_impact, sensor_budget,
                           pyomo_solver_options={'tee': False})
    expected_objective_value = 8760.59
    expected_selected_sensors = ["16", "21", "28", "38", "65",
                                 "__DUMMY_SENSOR_UNDETECTED__"]
    error = abs((results['objective_value'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
    
    use_prob = True
    solver = chama.optimize.Pmedian(scenario_prob=use_prob)
    results = solver.solve(df_sensor, df_scenario, df_impact, sensor_budget,
                           pyomo_solver_options={'tee': False})
    expected_objective_value = 9146.646
    expected_selected_sensors = ["16", "19", "38", "65", "68",
                                 "__DUMMY_SENSOR_UNDETECTED__"]
    error = abs((results['objective_value'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
