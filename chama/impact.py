"""
The impact module contains methods to extract the impact of detecting transport 
simulations given a set of defined sensor technologies.
"""
from __future__ import print_function
import pandas as pd
import numpy as np

def detection_times(signal, sensors, interp_method='linear', min_distance=10, metric=None):
    """
    Extract the impact metric from a signal profile and sensors

    Parameters
    ----------
    signal
    sensors
    metric
    interp_method
    min_distance

    Returns
    -------
    impact: pd.DataFrame
        DataFrame with columns 'Scenario', 'Sensor', and 'Impact'

    """
    # Extracting a subset of the signal in the sensor module is fastest
    # using multiindex even though setting the index initially is slow
    txyz = ['T', 'X', 'Y', 'Z']
    # check if the signal is already in multiindex form
    if not isinstance(signal.index, pd.MultiIndex):
        signal = signal.set_index(txyz)

    temp_det_times = {'Scenario': [], 'Sensor': [], 'T': []}

    for (name, sensor) in sensors.items():  # loop over sensors

        # Get detected signal
        detected = sensor.get_detected_signal(signal, interp_method,
                                              min_distance)

        # If the sensor detected something
        if detected.shape[0] > 0:
            for scenario_name, group in detected.groupby(level=[1]):
                temp_det_times['Scenario'].append(scenario_name)
                temp_det_times['Sensor'].append(name)
                temp_det_times['T'].append(group.index.get_level_values(0).tolist())
            
    det_times = pd.DataFrame()
    det_times['Scenario'] = temp_det_times['Scenario']
    det_times['Sensor'] = temp_det_times['Sensor']
    det_times['T'] = temp_det_times['T']

    det_times = det_times.sort_values('Scenario')
    det_times = det_times.reset_index(drop=True)
    
    return det_times

def detection_time_stats(det_times, statistic='min'):

    impact_col_name = list(set(det_times.columns) - set(['Scenario', 'Sensor']))[0]
    
    det_t = det_times.copy()
    for index, row in det_t.iterrows():
        if statistic == 'min':
            row[impact_col_name] = np.min(row[impact_col_name])
    
    det_t = det_t.rename(columns={impact_col_name: statistic+impact_col_name})
    
    return det_t
    
def translate(det_t, damage):
    
    impact_col_name = list(set(det_t.columns) - set(['Scenario', 'Sensor']))[0]
    
    damage = damage.set_index('T')
    allT = list(set(det_t[impact_col_name]) | set(damage.index))
    damage = damage.reindex(allT)
    damage.apply(pd.Series.interpolate)
    
    det_damage = det_t.copy()
    for index, row in det_damage.iterrows():
        row[impact_col_name] = damage.loc[row[impact_col_name],row['Scenario']]
    
    det_damage = det_damage.rename(columns={impact_col_name: 'Damage'})
    
    return det_damage

def _detection_times_to_coverage(det_times):
    
    impact_col_name = list(set(det_times.columns) - set(['Scenario', 'Sensor']))[0]
    
    temp = {'Scenario': [], 'Sensor': [], impact_col_name: []}
    for index, row in det_times.iterrows():
        for t in row['T']:
            temp['Scenario'].append(str((t,row['Scenario'])))
            temp['Sensor'].append(row['Sensor'])
            temp[impact_col_name].append(0.0)
    coverage = pd.DataFrame()
    coverage['Scenario'] = temp['Scenario']
    coverage['Sensor'] = temp['Sensor']
    coverage['Cov'] = temp['Cov']
    coverage = coverage.sort_values('Scenario')
    coverage = coverage.reset_index(drop=True)
    
    return coverage
