import unittest
from os.path import abspath, dirname, join
import pandas as pd
from pandas.testing import assert_frame_equal
import six
import chama

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

# ToDo: Add tests that verify the dataframes passed in are the same after solve

class TestWaterExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_water_network_example(self):
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
    
        # Solve sensor placement
        sensor_budget = 5
        solver = chama.optimize.ImpactFormulation()
        results = solver.solve(impact=df_impact, sensor=df_sensor,
                               scenario=df_scenario,
                               sensor_budget=sensor_budget)
        expected_objective_value = 8655.80
        error = abs((results['Objective'] -
                     expected_objective_value) / expected_objective_value)
        self.assertLess(error, 0.01)  # 1% error
        
        expected_selected_sensors = ["16", "21", "28", "38", "65"]
        self.assertListEqual(results['Sensors'], expected_selected_sensors)
    
    
    def test_water_network_example_with_scenario_prob(self):
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
        df_scenario.at['165', 'Probability'] = \
            1.0 - sum(df_scenario.iloc[1:].Probability)
        # Changing the undetected impact of scenario 165 such that the scenario
        # is not detected when scenario probabilities are ignored but forces a
        # different selection of sensors when scenario probabilities are
        # incorporated. 
        df_scenario.at['165', 'Undetected Impact'] = 30000.0
        df_scenario.reset_index(inplace=True)
    
        # Solve sensor placement
        sensor_budget = 5
        use_prob = False
        solver = chama.optimize.ImpactFormulation()
        results = solver.solve(impact=df_impact, sensor=df_sensor,
                               scenario=df_scenario, sensor_budget=sensor_budget,
                               use_scenario_probability=use_prob)
        expected_objective_value = 8760.59
        expected_selected_sensors = ["16", "21", "28", "38", "65"]
        error = abs((results['Objective'] -
                     expected_objective_value) / expected_objective_value)
        self.assertLess(error, 0.01)  # 1% error
        self.assertListEqual(results['Sensors'], expected_selected_sensors)
        
        use_prob = True
        solver = chama.optimize.ImpactFormulation()
        results = solver.solve(impact=df_impact, sensor=df_sensor,
                               scenario=df_scenario, sensor_budget=sensor_budget,
                               use_scenario_probability=use_prob)
        expected_objective_value = 9146.646
        expected_selected_sensors = ["16", "19", "38", "65", "68"]
        error = abs((results['Objective'] -
                     expected_objective_value) / expected_objective_value)
        self.assertLess(error, 0.01)  # 1% error
        self.assertListEqual(results['Sensors'], expected_selected_sensors)
    
    
    def test_water_network_example_with_grouping_constraint(self):
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
        solver = chama.optimize.ImpactFormulation()
    
        model = solver.create_pyomo_model(impact=df_impact, sensor=df_sensor,
                                          scenario=df_scenario)
    
        solver.add_grouping_constraint(['15', '16', '17'], select=2)
        solver.add_grouping_constraint(['16', '17', '18'], max_select=1)
    
        solver.solve_pyomo_model(sensor_budget=sensor_budget)
        self.assertTrue(solver._solved)
    
        results = solver.create_solution_summary()
        expected_objective_value = 9400.531
        expected_selected_sensors = ["15", "16", "19", "38", "65"]
        error = abs((results['Objective'] -
                     expected_objective_value) / expected_objective_value)
        self.assertLess(error, 0.01)  # 1% error
        self.assertListEqual(results['Sensors'], expected_selected_sensors)

class TestDetectionTimeToCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_detection_times_to_coverage_time(self):
        scenario = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Undetected Impact': [48.0, 250.0, 100.0],
            'Probability': [0.1, 0.1, 0.8]})
        detection_times = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Sensor': ['A', 'A', 'B'],
            'Detection Times': [[2, 3, 4], [3], [4, 5]]})
    
        # test coverage with no probabilities
        coverage, new_scenario = \
            chama.impact.detection_times_to_coverage(
                detection_times=detection_times,
                coverage_type='scenario-time',
                scenario=scenario)
        
        new_scenario.rename(columns={'Scenario': 'Entity',
                                     'Probability': 'Weight'},
                            inplace=True)
        
        solver = chama.optimize.CoverageFormulation()
        results = solver.solve(coverage=coverage, entity=new_scenario,
                               sensor_budget=1)
        self.assertListEqual(results['Sensors'], ['A'])
        # should do the same
        results = solver.solve(coverage=coverage, sensor_budget=1)
        self.assertListEqual(results['Sensors'], ['A'])
    
        # test coverage with probabilities - should be 'B' since scenario S3 is so
        # much more likely
        coverage, new_scenario = \
            chama.impact.detection_times_to_coverage(
                detection_times=detection_times,
                scenario=scenario,
                coverage_type='scenario-time')
            
        new_scenario.rename(columns={'Scenario': 'Entity',
                                     'Probability': 'Weight'},
                            inplace=True)
        
        solver = chama.optimize.CoverageFormulation()
        results = solver.solve(coverage=coverage, entity=new_scenario,
                               sensor_budget=1, use_entity_weight=True)
        self.assertListEqual(results['Sensors'], ['B'])
    
    
    def test_detection_times_to_coverage_scenario(self):
        scenario = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Undetected Impact': [48.0, 250.0, 100.0],
            'Probability': [0.1, 0.1, 0.8]})
        detection_times = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Sensor': ['A', 'A', 'B'],
            'Detection Times': [[2, 3, 4], [3], [4, 5]]})
    
        # test coverage with no probabilities
        coverage = chama.impact.detection_times_to_coverage(
            detection_times=detection_times, coverage_type='scenario')
    
        solver = chama.optimize.CoverageFormulation()
        results = solver.solve(coverage=coverage, sensor_budget=1)
        self.assertListEqual(results['Sensors'], ['A'])
    
        # test coverage with probabilities - should be 'B' since scenario S3 is so
        # much more likely
        coverage, new_scenario = \
            chama.impact.detection_times_to_coverage(
                detection_times=detection_times,
                scenario=scenario,
                coverage_type='scenario-time')
        new_scenario.rename(columns={'Scenario': 'Entity',
                                     'Probability': 'Weight'},
                            inplace=True)
            
        solver = chama.optimize.CoverageFormulation()
        results = solver.solve(coverage=coverage, entity=new_scenario,
                               sensor_budget=1, use_entity_weight=True)
        self.assertListEqual(results['Sensors'], ['B'])


class TestCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_coverage_solver(self):
        coverage_dict = {'A': [1, 2, 3], 'B': [1, 2], 'C': [3, 5],
                         'D': [4, 5], 'E': [2]}
        coverage_dict_reform = {'Sensor': [], 'Coverage': []}
        for key, value in six.iteritems(coverage_dict):
            coverage_dict_reform['Sensor'].append(key)
            coverage_dict_reform['Coverage'].append(value)
        coverage = pd.DataFrame(coverage_dict_reform)
    
        # test basic solve - should choose A and D
        cov_opt = chama.optimize.CoverageFormulation()
        results = cov_opt.solve(coverage=coverage, sensor_budget=2, redundancy=0)
        self.assertListEqual(sorted(results['Sensors']), ['A', 'D'])
        self.assertAlmostEqual(results['FractionDetected'], 1.0, places=4)
    
        # test redundancy - should choose A and B
        results = cov_opt.solve(coverage=coverage, sensor_budget=2, redundancy=1)
        self.assertListEqual(sorted(results['Sensors']), ['A', 'B'])
        self.assertAlmostEqual(results['FractionDetected'], 0.4, places=4)
    
        # test sensor cost - should choose A and C
        sensor_dict = {'Sensor': ['A', 'B', 'C', 'D', 'E'],
                       'Cost': [100.0, 1000.0, 10.0, 10.0, 100.0]
                       }
        sensor = pd.DataFrame(sensor_dict)
        results = cov_opt.solve(coverage=coverage, sensor=sensor,
                                sensor_budget=115.0, redundancy=1,
                                use_sensor_cost=True)
        self.assertListEqual(sorted(results['Sensors']), ['A', 'C'])
        self.assertAlmostEqual(results['FractionDetected'], 0.2, places=4)
    
        # test additional entities - should choose A and C, but with
        # FractionDetected = 0.125
        entity_dict = {'Entity': [1, 2, 3, 4, 5, 6, 7, 8]}
        entity = pd.DataFrame(entity_dict)
        results = cov_opt.solve(coverage=coverage, sensor=sensor,
                                sensor_budget=115.0, entity=entity,
                                redundancy=1, use_sensor_cost=True)
        self.assertListEqual(sorted(results['Sensors']), ['A', 'C'])
        self.assertAlmostEqual(results['FractionDetected'], 0.125, places=4)
    
        # test entity weights - should choose A and C, with FractionDetected = 0.2
        entity_dict = {'Entity': [1, 2, 3, 4, 5],
                       'Weight': [1, 1, 5, 1, 1]
                       }
        entity = pd.DataFrame(entity_dict)
        results = cov_opt.solve(coverage=coverage, sensor_budget=2,
                                entity=entity, redundancy=1,
                                use_entity_weight=True)
    
        self.assertListEqual(sorted(results['Sensors']), ['A', 'C'])
        self.assertAlmostEqual(results['FractionDetected'], 0.2, places=4)

if __name__ == '__main__':
    unittest.main()
