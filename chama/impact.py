"""
The impact module contains methods to extract the impact of detecting transport 
simulations given a set of defined sensor techninologies.
"""
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import time
import itertools

def interpolate(df, sample_points, method='nearest'):
    """
    Interpolate multiindex in df using sample points
    
    Parameters
    -----------
    df : pd.DataFrame
        A Multiindex pandas DataFrame
    
    sample_points : list of tuples
    
    method : string
        'linear' or 'nearest'
    
    Returns
    ---------
    df : pd.DataFrame
        
    """
    # Find points that are in sample points that are NOT in the df index
    all_sample_points = sum(sample_points.values(), [])
    new_points = list(set(all_sample_points) - set(df.index))
    
    if len(new_points) > 0:
        multiindex = pd.MultiIndex.from_tuples(new_points, names=df.index.names)
        interp_signal = pd.DataFrame(index=multiindex, columns=df.columns)
        xi = np.asarray(new_points)
        points = np.array([x for x in df.index])
        for col in df.columns:
            values = df.loc[:,col].values 
            interp_signal[col] = griddata(points, values, xi, method=method)
            
        df = df.append(interp_signal)
    
    return df
    
def extract(signal, sensors, metric='Time', txyz_names=['T', 'X', 'Y', 'Z']):
    """
    Extract the impact metric from a signal profile and sensors
    """  
    # Convert signal to a multiindex
    signal = signal.set_index(txyz_names)
    
    print("    Get sample points")
    t0 = time.time()
    sample_points = {}
    for (name, sensor) in sensors.items(): # loop over sensors
        sample_points[name] = sensor.get_sample_points()
    print(time.time() - t0)
    
    # Interpolate signal (if needed)
    print("    Interpolate")
    t0 = time.time()
    signal = interpolate(signal, sample_points)
    print(time.time() - t0)
    
    impact = pd.DataFrame(columns=['Scenario', 'Sensor', 'Impact'])
    
    extract_time = 0
    integrate_time = 0 
    print("    Extract/Integrate")
    for (name, sensor) in sensors.items(): # loop over sensors
        # Extract detected scenarios
        t0 = time.time()

        detected = signal.ix[sample_points[name]]
        extract_time = extract_time + (time.time() - t0)
        
        # Integrate
        t0 = time.time()
        detected = sensor.integrate_detected_signal(detected)
        integrate_time = integrate_time + (time.time() - t0)
        
        # Apply threshold
        detected = detected[detected > sensor.threshold]
        
        # Drop Nan and stack by index
        detected = detected.stack()
        
        # If the sensor detected something
        if detected.shape[0] > 0:
            if metric == 'Coverage':
                for row in range(detected.shape[0]):
                    col = str(detected.index[row])
                    val = 0
                    impact = impact.append({'Scenario': col, 'Sensor': name, 'Impact': val}, ignore_index=True)
            elif metric == 'Time':
                for col in set(detected.index.get_level_values(1)):
                    temp = detected.loc[:,col] # this returns a series
                    val = temp.index.min()
                    impact = impact.append({'Scenario': col, 'Sensor': name, 'Impact': val}, ignore_index=True)
    
    impact = impact.sort_values('Scenario')
    impact = impact.reset_index(drop=True)
    
    print(extract_time)
    print (integrate_time)
    
    return impact

def _translate(impact, metric):
    """
    Translate the time of detection to a different metric
    """
    # Gather sample points
    scenario = impact.index.get_level_values(0)
    time = impact.tolist()
    sample_points = list(zip(scenario,time))
    
    # Interpolate metrics (if needed)
    metric = interpolate(metric, sample_points)
    
    # There is probably a better way to do this without a loop
    data = {}
    for i in range(impact.shape[0]):
        data[(impact.index[i][0], impact.index[i][1])] = metric[impact.index[i][0]][impact.iloc[i]]
    
    # Reindex
    impact_metric = pd.Series(data)
    impact_metric.index.names = ['Scenario', 'Sensor']
    
    return impact_metric

#def _add_nondetected_impact(impact_metric, scenario_list, nondetected_impact_value):
#    multiindex = pd.MultiIndex.from_tuples(list(zip(scenario_list, len(scenario_list)*['_NotDetected'])), names=impact_metric.index.names)
#    nondetected_impact = pd.DataFrame(index=multiindex, data=nondetected_impact_value, columns=impact_metric.columns)
#    impact_metric = impact_metric.append(nondetected_impact)
#
#    return impact_metric

