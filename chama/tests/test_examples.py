from nose.tools import *
from nose.plugins.skip import SkipTest
from os.path import abspath, dirname, join
import chama
import pandas as pd
import numpy as np

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

def test_simple_example():

    # Skipping this test until it can be replaced with a realistic
    # example
    raise SkipTest

    # Read signal and metrics file (generated from simualtions)
    signal_file = join(datadir, 'signals.csv')
    signal = pd.read_csv(signal_file)
    t_col = 'timedatestamp'
    x_col = 'x-coord'
    y_col = 'y-coord'
    z_col = 'z-coord'
    sim_times = sorted(set(signal.loc[:, t_col]))

    signal.rename(index=str, columns={t_col: 'T', x_col: 'X',
                                      y_col: 'Y', z_col: 'Z'})

    sensors = {}
    i = 0
    # Define sensors
    for x in np.arange(-700, 1000, 100): 
        for y in np.arange(-200, 1000, 100):
            z = 3
            i += 1
            sensor = chama.sensors.Sensor(location=(x, y, z),
                                          threshold=100,
                                          sample_times=list(sim_times))
            name = 'Point Sensor ' + str(i)
            sensors[name] = sensor
                     
    # Compute impact
    df_impact = chama.impact.extract(signal, sensors, metric='Count',
                                     interp_method='nearest')
    
    # Define sensor dataframe
    df_sensor = pd.DataFrame(data={'Sensor': list(sensors.keys())})
    df_sensor['Cost'] = 1.0     # define the cost
    
    # Define scenario dataframe
    df_scenario = pd.DataFrame(data=df_impact['Scenario'].unique(),
                               columns=['Scenario'])
    df_scenario['Undetected Impact'] = 1.0  
    
    # Solve sensor placement
    sensor_budget = 5
    solver = chama.optimize.Pmedian()
    results = solver.solve(df_sensor, df_scenario, df_impact,
                           sensor_budget, pyomo_solver_options={'tee': False})
    
    expected_selected_sensors = ['Point Sensor 88', 'Point Sensor 89',
                                 'Point Sensor 90', 'Point Sensor 91',
                                 'Point Sensor 99']
    assert_list_equal(results['selected_sensors'], expected_selected_sensors)
