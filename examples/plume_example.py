"""
The following example uses Chama to optimize the placement of sensors that 
detects a set of potential plumes from 10 well pads based on 1 hour wind data. 
Simulation data is created using the Gaussian plume model in Chama.
The coverage formulation is used to optimize sensor placement.
Note that this example uses notional wind data, leak rates, and sensor thresholds
along with low resolution simulations for demonstration purposes.
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import chama

### Define input
np.random.seed(123)
wellpads = pd.read_csv('wellpads.csv')
wind_data = pd.read_csv('wind_data_1hr.csv', index_col='Time') 
nRealizations = 10
xsize = 100 # m
ysize = 100 # m
zsize = 10 # m
tsize = 24*60*60 # s (1 day)
dx = 10 # m
dy = 10 # m
dz = 1 # m
dt = 3600 # s (hourly)

### Build the x,y,z,t grid
xar = np.arange(0, xsize, dx)
yar = np.arange(0, ysize, dy)
zar = np.arange(0, zsize, dz)
tar = np.arange(0, tsize, dt)
tsteps = int(tsize/dt)
grid = chama.simulation.Grid(xar, yar, zar)

### Generate plume signal data (for each wellpad and realization)
signal = None
info = []
for index, wellpad in wellpads.iterrows():
    x = wellpad['X']
    y = wellpad['Y']
    z = 0
    for realization in range(nRealizations):
        scenario_name = 'Well' + str(index)+'_'+str(realization)
        print(scenario_name)
        leakrate = np.random.uniform(low=0.001, high=0.1)
        info.append([scenario_name, leakrate, x, y, z])
        
        source = chama.simulation.Source(x, y, z, leakrate)
        gauss_plume = chama.simulation.GaussianPlume(grid, source, wind_data)
        gauss_plume.run()
        conc = gauss_plume.conc
        # rename the scenario S to a unique scenario name
        conc = conc.rename(columns={'S': scenario_name})
        if signal is None:
            signal = conc
        else:
            signal[scenario_name] = conc[scenario_name]
            
signal.to_csv('signal.csv')
signal_info = pd.DataFrame(info, columns=['Scenario', 'Leakrate', 'X', 'Y', 'Z'])

### Define sensors   
sensors = {}
threshold = 1e-4
sensor_num = 0
for x in np.arange(0,100,10):
    for y in np.arange(0,100,10):
        for z in [0,1,2,3,4,5,6,7,8,9]:
            sensor_location = (x, y, z)
            sensor_num += 1
            loc = chama.sensors.Stationary(location=sensor_location)
            pt = chama.sensors.Point(threshold=threshold, sample_times=tar)
            det = chama.sensors.Sensor(position=loc, detector=pt)
            sensor_name = 'Sensor'+str(sensor_num)
            sensors[sensor_name] = det

### Impact assessment
det_times = chama.impact.extract_detection_times(signal, sensors)
scenario_coverage  = chama.impact.detection_times_to_coverage(det_times, coverage_type='scenario')

### Sensor placement
coverage = chama.optimize.CoverageFormulation()
results = coverage.solve(scenario_coverage, sensor_budget=5,
                         coverage_col_name='Coverage', mip_solver_name='glpk')
    
### Graphics
selected_sensors = dict([(s, sensors[s]) for s in results['Sensors']])
chama.graphics.sensor_locations(selected_sensors, x_range=(0,100),
                                y_range=(0,100), z_range=(0,10))
ax = plt.gca()
for ndex, wellpad in wellpads.iterrows():
    ax.scatter(wellpad['X'], wellpad['Y'],0,c='k',marker='x')

# Simple test to ensure results don't change
assert results['Objective'] == 99.0
assert results['Sensors'] == ['Sensor384', 'Sensor551', 'Sensor575', 'Sensor731', 'Sensor797']