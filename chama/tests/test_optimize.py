from nose.tools import *
from nose.plugins.skip import SkipTest
from os.path import abspath, dirname, join
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
import chama
import pyomo.environ as pe

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
    results = solver.solve(df_impact, df_sensor, df_scenario, sensor_budget=sensor_budget,
                           pyomo_solver_options={'tee': False})

    expected_objective_value = 8655.80
    error = abs((results['Objective'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    
    expected_selected_sensors = ["16", "21", "28", "38", "65"]
    assert_list_equal(results['Sensors'], expected_selected_sensors)
    

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
    solver = chama.optimize.Pmedian()
    results = solver.solve(df_impact, df_sensor, df_scenario, sensor_budget=sensor_budget,
                           use_scenario_probability=use_prob, pyomo_solver_options={'tee': False})
    expected_objective_value = 8760.59
    expected_selected_sensors = ["16", "21", "28", "38", "65"]
    error = abs((results['Objective'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    assert_list_equal(results['Sensors'], expected_selected_sensors)
    
    use_prob = True
    solver = chama.optimize.Pmedian()
    results = solver.solve(df_impact, df_sensor, df_scenario,sensor_budget=sensor_budget,
                           use_scenario_probability=use_prob, pyomo_solver_options={'tee': False})
    expected_objective_value = 9146.646
    expected_selected_sensors = ["16", "19", "38", "65", "68"]
    error = abs((results['Objective'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    assert_list_equal(results['Sensors'], expected_selected_sensors)


def test_water_network_example_with_grouping_constraint():
    impact_file = join(datadir, 'Net3_ec.impact')

    # read the impact file from the water simulations
    impact_data = pd.read_csv(impact_file, skiprows=2, sep=' ',
                              usecols=[0, 1, 3],
                              names=['Scenario', 'Sensor', 'Impact'])
    impact_data['Scenario'] = impact_data['Scenario'].apply(
        str)  # convert the scenario names to strings
    impact_data['Sensor'] = impact_data['Sensor'].apply(
        str)  # convert the Sensor names to strings

    # Define sensor dataframe
    df_sensor = pd.DataFrame(data=impact_data['Sensor'].unique(),
                             columns=['Sensor'])
    df_sensor = df_sensor[df_sensor.Sensor != "-1"]
    df_sensor['Cost'] = 1.0  # define the cost

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

    model = solver.create_pyomo_model(df_impact, df_sensor, df_scenario, sensor_budget=sensor_budget)

    solver.add_grouping_constraint(['15', '16', '17'], select=2)
    solver.add_grouping_constraint(['16', '17', '18'], max_select=1)

    solver_results = solver.solve_pyomo_model()

    results = solver.create_solution_summary()
    expected_objective_value = 9400.531
    expected_selected_sensors = ["15", "16", "19", "38", "65"]
    error = abs((results['Objective'] -
                 expected_objective_value) / expected_objective_value)
    assert_less(error, 0.01)  # 1% error
    assert_list_equal(results['Sensors'], expected_selected_sensors)

def test_detection_times_to_coverage_time():
    scenario = pd.DataFrame({
        'Scenario': ['S1', 'S2', 'S3'],
        'Undetected Impact': [48.0, 250.0, 100.0],
        'Probability': [0.25, 0.60, 0.15]})
    impact = pd.DataFrame({
        'Scenario': ['S1', 'S2', 'S3'],
        'Sensor': ['A', 'A', 'B'],
        'Impact': [[2, 3, 4], [3], [4, 5]]})

    coverage = chama.optimize.Coverage()
    impact1,scenario1 = coverage.convert_detection_times_to_coverage(impact, scenario,
                                                                    use_scenario_probability=True,
                                                                     coverage_type='scenario-time')
    impact_expected = pd.DataFrame([("(2, 'S1')", 'A', 0.0),
                                    ("(3, 'S1')", 'A', 0.0),
                                    ("(3, 'S2')", 'A', 0.0),
                                    ("(4, 'S1')", 'A', 0.0),
                                    ("(4, 'S3')", 'B', 0.0),
                                    ("(5, 'S3')", 'B', 0.0)],
                                columns=['Scenario', 'Sensor', 'Impact'])  
    sceanrio_expected = pd.DataFrame([("(2, 'S1')", 1.0, 0.25),
                                      ("(3, 'S1')", 1.0, 0.25),
                                      ("(3, 'S2')", 1.0, 0.6),
                                      ("(4, 'S1')", 1.0, 0.25),
                                      ("(4, 'S3')", 1.0, 0.15),
                                      ("(5, 'S3')", 1.0, 0.15)],
                                columns=['Scenario', 'Undetected Impact', 'Probability'])
    
    impact1.set_index('Scenario', inplace=True)
    impact_expected.set_index('Scenario', inplace=True)
    assert_frame_equal(impact1, impact_expected, check_dtype=False,
                           check_like=True)
    
    scenario1.set_index('Scenario', inplace=True)
    sceanrio_expected.set_index('Scenario', inplace=True)
    assert_frame_equal(scenario1, sceanrio_expected, check_dtype=False,
                           check_like=True)

def test_detection_times_to_coverage_scenario():
    scenario = pd.DataFrame({
        'Scenario': ['S1', 'S2', 'S3'],
        'Undetected Impact': [48.0, 250.0, 100.0],
        'Probability': [0.25, 0.60, 0.15]})
    impact = pd.DataFrame({
        'Scenario': ['S1', 'S2', 'S3'],
        'Sensor': ['A', 'A', 'B'],
        'Impact': [[2, 3, 4], [3], [4, 5]]})

    coverage = chama.optimize.Coverage()
    impact1,scenario1 = coverage.convert_detection_times_to_coverage(impact, scenario,
                                                              use_scenario_probability=True,
                                                              coverage_type='scenario'
                                                              )
    impact_expected = pd.DataFrame([('S1', 'A', 0.0),
                                    ('S2', 'A', 0.0),
                                    ('S3', 'B', 0.0)],
                                columns=['Scenario', 'Sensor', 'Impact'])  
    sceanrio_expected = pd.DataFrame([('S1', 1.0, 0.25),
                                      ('S2', 1.0, 0.6),
                                      ('S3', 1.0, 0.15)],
                                columns=['Scenario', 'Undetected Impact', 'Probability'])
    
    impact1.set_index('Scenario', inplace=True)
    impact_expected.set_index('Scenario', inplace=True)
    assert_frame_equal(impact1, impact_expected, check_dtype=False,
                           check_like=True)
    
    scenario1.set_index('Scenario', inplace=True)
    sceanrio_expected.set_index('Scenario', inplace=True)
    assert_frame_equal(scenario1, sceanrio_expected, check_dtype=False,
                           check_like=True)

        

if __name__ == '__main__':
#    test_water_network_example()
    test_detection_times_to_coverage_scenario()
