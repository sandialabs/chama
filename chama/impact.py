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
        Signal data from the transport simulation.  The DataFrame has columns  
        'T', 'X','Y', 'Z', or 'T', 'Node' along with one column for each 
        scenario (user defined names).
    
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
    pandas DataFrame with columns 'Scenario', 'Sensor', and 'Detection Times'.
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
        
    det_times_dict = {'Scenario': [], 'Sensor': [], 'Detection Times': []}

    for (name, sensor) in sensors.items():  # loop over sensors
        # Get detected signal
        detected = sensor.get_detected_signal(signal, interp_method,
                                              min_distance)

        # If the sensor detected something
        if detected.shape[0] > 0:
            for scenario_name, group in detected.groupby(level=[1]):
                det_times_dict['Scenario'].append(scenario_name)
                det_times_dict['Sensor'].append(name)
                det_times_dict['Detection Times'].append(group.index.get_level_values(0).tolist())
            
    det_times = pd.DataFrame(det_times_dict)
    det_times = det_times[['Scenario', 'Sensor', 'Detection Times']] # reorder

    det_times = det_times.sort_values(['Scenario', 'Sensor'])
    det_times = det_times.reset_index(drop=True)
    
    return det_times


def detection_time_stats(det_times):
    """
    Returns detection times statistics (min, mean, median, max, and count).

    Parameters
    ----------
    det_times : pandas DataFrame
        Detection times for each scenario-sensor pair. The DataFrame has 
        columns  'Scenario', 'Sensor', and 'Detection Times', see 
        :class:`~chama.impact.detection_times`.
        
    Returns
    ----------
    pandas DataFrame with columns 'Scenario', 'Sensor', 'Min', 'Mean', 
    'Median', 'Max', and 'Count'.  
    """
    det_t = det_times.copy()
    det_t['Min'] = det_t['Detection Times'].apply(np.min)
    det_t['Mean'] = det_t['Detection Times'].apply(np.mean)
    det_t['Median'] = det_t['Detection Times'].apply(np.median)
    det_t['Max'] = det_t['Detection Times'].apply(np.max)
    det_t['Count'] = det_t['Detection Times'].apply(len)
    
    del det_t['Detection Times']
    
    return det_t

def translate(det_t, damage):
    """
    Returns impact translated from detection time to a damage metric.
    
    Parameters
    ----------
    det_t : pandas DataFrame
        Detection time for each scenario-sensor pair. The DataFrame has 
        columns 'Scenario', 'Sensor', and 'T'. 
    damage : pandas DataFrame
        Damage values for each scenario and time.  The DataFrame has 
        columns 'T' and one column for each scenario (user defined names).
        
    Returns
    ----------
    pandas DataFrame with columns 'Scenario', 'Sensor', and 'Damage'.  
    """
    damage = damage.set_index('T')
    allT = list(set(det_t['T']) | set(damage.index))
    damage = damage.reindex(allT)
    damage.sort_index(inplace=True)
    damage.interpolate(inplace=True)
    
    det_damage = det_t.copy()
    det_damage['T'] = damage.lookup(det_t['T'], det_t['Scenario'])
    det_damage.rename(columns = {'T':'Damage'}, inplace = True)

    return det_damage
