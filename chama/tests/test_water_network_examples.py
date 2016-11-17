from nose.tools import *
from os.path import abspath, dirname, join
import chama
import pandas as pd

testdir = dirname(abspath(__file__))
datadir = join(testdir,'data')

def test_net3_ec():
    # This test replicates WST sp_ex1
    impact_file = join(datadir, 'Net3_ec.impact')
    
    impact_data =  pd.read_csv(impact_file, skiprows=2, sep =' ', usecols=[0,1,3]) 
    impact_data.columns=['Scenario','Sensor','Impact']
    
    # Define sensors  
    # This will be replaced by a dictonary that defines sensor cost
    sensor_names = set(impact_data['Sensor'])
    sensors = dict()
    for i in sensor_names:
        sensors[i] = chama.sensors.Point()
    
    # Solve sensor placement
    sensor_budget = 5
    solver = chama.design.PyomoDesignSolver('glpk')
    results = solver.solve(impact_data, sensors, sensor_budget)
    
    expected_objective_value = 8655.80
    error = abs((results['objective_value'] - expected_objective_value)/expected_objective_value)
    assert_less(error, 0.01) # 1% error
    
    expected_selected_sensors = [16, 21, 28, 38, 65, -1]
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
    
if __name__ == '__main__':
    test_net3_ec()