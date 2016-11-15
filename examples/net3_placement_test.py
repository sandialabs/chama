import chama
import numpy as np
import pandas as pd

# Read data from impact file
impact_data =  pd.read_csv('Data/Net3_ec.impact', skiprows=2, sep =' ', usecols=[0,1,3], dtype={0: np.str_})
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

print(results)