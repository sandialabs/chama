"""
The impact module contains methods to extract the impact of detecting transport
simulations given a set of defined sensor technologies.
"""
from __future__ import print_function, division
import pandas as pd
import numpy as np


def detection_times(signal, sensors, interp_method=None, min_distance=10):
    """
    Returns detection times from a signal and group of sensors.

    Parameters
    ----------
    signal : pandas DataFrame
        Signal data from the transport simulation
    
    sensors : dict
        A dictionary of sensors with key:value pairs containing
        {'sensor name': chama Sensor object}
    
    interp_method : 'linear', 'nearest', or None
        Method used to interpolate the signal if needed.  
        A value of 'linear' will use griddata to interpolate missing
        sample points. A value of 'nearest' will set the sample point to
        the nearest signal point within a minimum distance of min_distance.
        If there are no signal points within this distance then the
        signal will be set to zero at the sample point.

    min_distance : float
        The minimum distance when using the 'nearest' interp_method
    
    Returns
    -------
    pandas DataFrame with columns 'Scenario', 'Sensor', and 'Impact'.  
    The Impact column contains a list of detection times.
    """
    # Extracting a subset of the signal in the sensor module is fastest
    # using multiindex even though setting the index initially is slow
    # check if the signal is already in multiindex form 
    if not isinstance(signal.index, pd.MultiIndex):
        if set(['T', 'X', 'Y', 'Z']) < set(list(signal.columns)):
            signal = signal.set_index(['T', 'X', 'Y', 'Z'])
        elif set(['T', 'Node']) < set(list(signal.columns)):
            signal = signal.set_index(['T', 'Node'])
        else:
            raise ValueError('Unrecognized signal format')
            return
    
    if 'Node' in signal.index.names:
        interp_method = None
        
    temp_det_times = {'Scenario': [], 'Sensor': [], 'Impact': []}

    for (name, sensor) in sensors.items():  # loop over sensors
        # Get detected signal
        detected = sensor.get_detected_signal(signal, interp_method,
                                              min_distance)

        # If the sensor detected something
        if detected.shape[0] > 0:
            for scenario_name, group in detected.groupby(level=[1]):
                temp_det_times['Scenario'].append(scenario_name)
                temp_det_times['Sensor'].append(name)
                temp_det_times['Impact'].append(group.index.get_level_values(0).tolist())
            
    det_times = pd.DataFrame()
    det_times['Scenario'] = temp_det_times['Scenario']
    det_times['Sensor'] = temp_det_times['Sensor']
    det_times['Impact'] = temp_det_times['Impact']

    det_times = det_times.sort_values(['Scenario', 'Sensor'])
    det_times = det_times.reset_index(drop=True)
    
    return det_times


def detection_time_stats(det_times):
    """
    Returns detection times statistics (min, max, median, and mean).

    Parameters
    ----------
    det_times : pandas DataFrame
        Detection times returned from detection_times
        
    Returns
    ----------
    pandas DataFrame with columns 'Scenario', 'Sensor', 'Min', 'Max', 'Median' and 'Mean'.  
    """
    det_t = det_times.copy()
    det_t['Min'] = det_t['Impact'].apply(np.min)
    det_t['Max'] = det_t['Impact'].apply(np.max)
    det_t['Median'] = det_t['Impact'].apply(np.median)
    det_t['Mean'] = det_t['Impact'].apply(np.mean)
    del det_t['Impact']
    
    return det_t
 
def translate(det_t, damage):
    """
    Returns impact translated from detection time to a damage metric.
    
    Parameters
    ----------
    det_t : pandas DataFrame
        Detection times returned from detection_time_stats
    damage : pandas DataFrame
        Damage values for each scenario and time.  The DataFrame has 
        columns 'T' and one column for each scenario (user defined names)
        
    Returns
    ----------
    pandas DataFrame with columns 'Scenario', 'Sensor', and 'Impact'.  
    The Impact column contains damage at the time of detection.
    """
    damage = damage.set_index('T')
    allT = list(set(det_t['Impact']) | set(damage.index))
    damage = damage.reindex(allT)
    damage.apply(pd.Series.interpolate)
    
    det_damage = det_t.copy()
    for index, row in det_damage.iterrows():
        det_damage.loc[index, 'Impact'] = damage.loc[row['Impact'],
                                                     row['Scenario']]

    return det_damage
