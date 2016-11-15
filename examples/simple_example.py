import chama
import pandas as pd
import numpy as np
import time

t0 = time.time()

multiindex=True

# Read signal and metrics file (generated from simualtions)
signal = pd.read_csv('data/AERMOD_SIGNAL1-7.csv')
#signal = pd.read_csv('data/CALPUFF_SIGNAL1-7.csv')
t_col = 'timedatestamp'
x_col = 'x-coord'
y_col = 'y-coord'
z_col = 'z-coord'
sim_times = sorted(set(signal.loc[:,t_col]))
    
sensors = {}
i = 0
print("Define sensors")
np.random.seed(100)
for x in np.arange(-700, 1000, 100): 
    for y in np.arange(-200, 1000, 100):
        z = 3
        i = i+1
        #x = np.random.rand()*1700-700
        #y = np.random.rand()*1200-200
        point_sensor = chama.sensors.Point()
        point_sensor.location = [x,y,z]
        point_sensor.threshold = 100 # units of signal
        point_sensor.sampling_times = list(sim_times)  # units of T
        name = 'Point Sensor ' + str(i)
        sensors[name] = point_sensor
            
print("Compute impact")
impact_data = chama.impact.extract(signal, sensors, metric='Count', txyz_names=[t_col,x_col,y_col,z_col])
print(impact_data)
print(time.time() - t0)

print("Run Optimization")
if impact_data.shape[0] > 0:
    sensor_budget = 5
    solver = chama.design.PyomoDesignSolver('glpk')
    results = solver.solve(impact_data, sensors, sensor_budget)
else:
    results = {'selected_sensors': []}

print(results)