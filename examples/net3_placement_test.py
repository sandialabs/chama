import chama
import numpy as np
import pandas as pd

# Read data from impact file
df =  pd.read_csv('Data/Net3_ec.impact', skiprows=2, sep =' ', usecols=[0,1,3], index_col=[0,1], dtype={0: np.str_})
index = pd.MultiIndex.from_tuples(df.ix[:,0].index.tolist(),names=['Scenario','Sensor'])
impact_data = pd.Series(data=df.ix[:,0].tolist(), index=index)

# Define sensors  
sensor_names = set(impact_data.index.levels[1])
sensors = dict()
for i in sensor_names:
    sensors[i] = chama.sensors.Point()

# Solve sensor placement
sensor_budget = 5
solver = chama.design.PyomoDesignSolver('glpk')
sp_results = solver.solve(impact_data, sensors, sensor_budget)

print(sp_results)