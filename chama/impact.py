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


def detection_times_to_coverage(detection_times, coverage_type='scenario'):
    """
    Convert a detection times DataFrame to a coverage DataFrame for input to a coverage-based sensor placement solver.
    
    Parameters
    ----------
    detection_times : pandas DataFrame
        DataFrame containing three columns. 'Scenario' is the name of the scenarios, 'Sensor' is the name of 
        the sensors, and 'Detection Times' contains a list of the detection times.
    coverage_type : str
        Sets the coverage type: 'scenario' builds a coverage matrix designed to ensure coverage of the scenario
        ignoring the time it was detected, while 'scenario-time' builds a coverage matrix where every scenario-time
        pair is included as a new scenario - thereby ensuring coverage over all scenarios and times.

    Returns
    -------
    DataFrame : coverage DataFrame to be used an input to a coverage-based sensor placement solver.

    """
    if coverage_type == 'scenario':
        coverage = det_times
        # remove any entries where detection times is an empty list
        coverage = coverage[coverage.apply(lambda x: len(x['Detection Times']) != 0, axis=1)]
        # drop the detection times
        coverage.drop('Detection Times', axis=1, inplace=True)
    else:  # coverage_type=='scenario-time':
        # Add scenario probability to det_times
        det_times.set_index('Scenario')
        scenario.set_index('Scenario')
        det_times['Probability'] = scenario['Probability']
        det_times.reset_index(drop=True)

            # To avoid strange behavoir in df.apply, add a dummy first row
            # that has 1 value for Impact
            dummy = pd.DataFrame({
                'Scenario': ['dummy'],
                'Sensor': ['dummy'],
                'Impact': [[0]]})
            det_times = pd.concat([dummy, det_times], ignore_index=True)

            # Expand times
            times = list(itertools.chain.from_iterable(det_times['Impact'].values))

            def expand_values(row, col_name):
                return [row[col_name]] * len(row['Impact'])

            # Expand scenarios
            scenarios = det_times.apply(expand_values, col_name='Scenario', axis=1)
            scenarios = list(itertools.chain.from_iterable(scenarios.values))

            # Expand sensors
            sensors = det_times.apply(expand_values, col_name='Sensor', axis=1)
            sensors = list(itertools.chain.from_iterable(sensors.values))

            # Expand probabilities
            if self.use_scenario_probability:
                probability = det_times.apply(expand_values, col_name='Probability', axis=1)
                probability = list(itertools.chain.from_iterable(probability.values))

            # Updated scenario dataframe
            scenario = pd.DataFrame({'Scenario': list(zip(times, scenarios))})
            if self.use_scenario_probability:
                scenario['Probability'] = probability
            scenario.drop(0, inplace=True)  # drop dummy
            scenario = scenario.sort_values('Scenario')
            scenario = scenario.reset_index(drop=True)
            scenario['Scenario'] = scenario['Scenario'].apply(str)

            # Updated impact dataframe
            coverage = pd.DataFrame({'Scenario': list(zip(times, scenarios)),
                                     'Sensor': sensors})
            coverage.drop(0, inplace=True)  # drop dummy
            coverage = coverage.sort_values('Scenario')
            coverage = coverage.reset_index(drop=True)
            coverage['Scenario'] = coverage['Scenario'].apply(str)

        coverage['Impact'] = 0.0
        scenario['Undetected Impact'] = 1.0

        return coverage, scenario


def impact_to_coverage():
    CRASH HERE!


