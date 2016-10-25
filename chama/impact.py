"""
Impact assessment .
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
    new_points = list(set(sample_points) - set(df.index))
    
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
    
def interpolate2(df, sample_points, method='nearest', txyz_names=['T', 'X', 'Y', 'Z']):
    # Extract TXYZ as a numpy array
    signal_txyz = df.loc[:,txyz_names].as_matrix()
    
    # Find points that are in sample points that are NOT in signal_txyz
    new_points = np.array([x for x in set(sample_points) - set(tuple(x) for x in signal_txyz)])
    
    if new_points.shape[0] > 0:
        s_cols = [col for col in df.columns if col not in txyz_names]
        interp_signal = pd.DataFrame(data=new_points, columns=txyz_names)
        for col in s_cols:
            interp_values = griddata(signal_txyz, df.loc[:,col].values, new_points, method=method)
            interp_signal[col] = interp_values
    
        df = df.append(interp_signal)
    
    return df
    
def detection_time(signal, sensors, metric='Min Time', multiindex=True, txyz_names=['T', 'X', 'Y', 'Z']):
    """
    Extract the time of detection from a signal profile and sensors
    """    
    # Gather sample points
    print("    Get sample points")
    t0 = time.time()
    all_sample_points = []
    sample_points = {}
    for (name, sensor) in sensors.items(): # loop over sensors
        sample_points[name] = sensor.get_sample_points()
        all_sample_points = all_sample_points + sample_points[name]
    all_sample_points = list(set(all_sample_points))
    print(time.time() - t0)
    
    # Interpolate signal (if needed)
    print("    Interpolate")
    t0 = time.time()
    if multiindex:
        signal = interpolate(signal, all_sample_points)
    else:
        signal = interpolate2(signal, all_sample_points, txyz_names=txyz_names)
    print(time.time() - t0)
    
    impact = pd.DataFrame(columns=['Scenario', 'Sensor', 'Impact'])
    if metric == 'Count':
        sim_times = sorted(signal.index.get_level_values(txyz_names[0]).unique())
        run_names = sorted(list(signal.columns.values))
        scenarios = list(str(i) for i in itertools.product(sim_times, run_names))
        d = {'Scenario': scenarios, 'Sensor': '_NotDetected', 'Impact': 1.0}
        nondetected_impact = pd.DataFrame(d)
        impact = impact.append(nondetected_impact)
    
    extract_time = 0
    integrate_time = 0 
    print("    Extract/Integrate")
    for (name, sensor) in sensors.items(): # loop over sensors
        # Extract detected scenarios
        t0 = time.time()

        if multiindex:
            detected = signal.ix[sample_points[name]]
        else:
            ## HOW DO YOU DO THIS WITHOUT CONVERTING TO A MUTLIINDEX DATAFRAME?
            # use df1.isin(df2)?
            signal = signal.set_index(txyz_names)
            detected = signal.ix[sample_points[name]]
            signal = signal.reset_index()
            detected = detected.reset_index()
        extract_time = extract_time + (time.time() - t0)
        
        # Integrate
        t0 = time.time()
        if multiindex:
            detected = sensor.integrate_detected_signal(detected)
        else:
            detected = sensor.integrate_detected_signal2(detected, txyz_names=txyz_names) 
        integrate_time = integrate_time + (time.time() - t0)
        
        # Apply threshold
        detected = detected[detected > sensor.threshold]
        
        # Drop Nan and stack by index
        detected = detected.stack()
        
        # If the sensor detected something
        if detected.shape[0] > 0:
            for row in range(detected.shape[0]):
                col = str(detected.index[row])
                if metric == 'Count':
#                    val = 1
                    val = 0
                elif metric == 'Signal':
                    val = detected.iloc[row]
                    
                impact = impact.append({'Scenario': col, 'Sensor': name, 'Impact': val}, ignore_index=True)
                
#            for col in set(detected.index.get_level_values(1)):
#                temp = detected.loc[:,col] # this returns a series
#                
#                if metric == 'Min Time':
#                    val = temp.index.min()
#                elif metric == 'Max Time':
#                    val = temp.index.min()
#                elif metric == 'Min Signal':
#                    val = temp.min()
#                elif metric == 'Max Signal':
#                    val = temp.max()
#                elif metric == 'Count':
#                    val = temp.shape[0]
#
#                impact = impact.append({'Scenario': col, 'Sensor': name, 'Impact': val}, ignore_index=True)
    
    impact = impact.sort_values('Scenario')
    
    
    if multiindex:
        impact = impact.set_index(['Scenario', 'Sensor'])
    else:
        impact = impact.reset_index(drop=True)
        
    print(extract_time)
    print (integrate_time)
    
    return impact

def translate(impact_time, metric):
    """
    Translate the time of detection to a different metric
    """
    # Gather sample points
    scenario = impact_time.index.get_level_values(0)
    time = impact_time.tolist()
    sample_points = list(zip(scenario,time))
    
    # Interpolate metrics (if needed)
    metric = interpolate(metric, sample_points)
    
    # There is probably a better way to do this without a loop
    data = {}
    for i in range(impact_time.shape[0]):
        data[(impact_time.index[i][0], impact_time.index[i][1])] = metric[impact_time.index[i][0]][impact_time.iloc[i]]
    
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

