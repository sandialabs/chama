"""
The impact module contains methods to extract detection times from a set of
simulations and sensor technologies along with methods to convert detection 
times to impact and coverage metrics.

.. rubric:: Contents

.. autosummary::

    extract_detection_times
    detection_time_stats
    detection_time_to_impact
    detection_times_to_coverage
    impact_to_coverage
"""
from __future__ import print_function, division
import pandas as pd
import numpy as np


def extract_detection_times(signal, sensors, interp_method=None,
                            min_distance=10.0):
    """
    Returns detection times from a signal and group of sensors.

    Parameters
    ----------
    signal : pandas DataFrame
        Signal data from the simulation.  The DataFrame can be in XYZ format 
        (with columns named 'X','Y','Z','T') or Node format (with columns named
        'Node','T') along with one column for each scenario (user defined
        names).
    
    sensors : dict
        A dictionary of sensors with key:value pairs containing
        {'sensor name': chama :py:class:`Sensor<chama.sensors.Sensor>` object}
    
    interp_method : 'linear', 'nearest', or None
        Method used to interpolate the signal if needed.  
        A value of 'linear' will use griddata to interpolate missing
        sample points. A value of 'nearest' will set the sample point to
        the nearest signal point within a minimum distance of min_distance.
        If there are no signal points within this distance then the
        signal will be set to zero at the sample point. Note that interpolation
        is not used when the signal is in Node format.

    min_distance : float
        The minimum distance when using the 'nearest' interp_method
    
    Returns
    -------
    det_times : pandas DataFrame
        DataFrame with columns 'Scenario', 'Sensor', and 'Detection Times'.
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
                det_times_dict['Detection Times'].append(
                    group.index.get_level_values(0).tolist())
            
    det_times = pd.DataFrame(det_times_dict)
    det_times = det_times[['Scenario', 'Sensor', 'Detection Times']]  # reorder

    det_times = det_times.sort_values(['Scenario', 'Sensor'])
    det_times = det_times.reset_index(drop=True)
    
    return det_times


def detection_time_stats(detection_times):
    """
    Returns detection times statistics (min, mean, median, max, and count).
    
    The minimum detection time is often used as input to an
    impact-based sensor placement solver.

    Parameters
    ----------
    detection_times : pandas DataFrame
        Detection times for each scenario-sensor pair. The DataFrame has
        columns  'Scenario', 'Sensor', and 'Detection Times', see
        :class:`~chama.impact.detection_times`.
        
    Returns
    ----------
    det_t : pandas DataFrame
        DataFrame with columns 'Scenario', 'Sensor', 'Min', 'Mean',
        'Median', 'Max', and 'Count'.
    """
    det_t = detection_times.copy()
    det_t['Min'] = det_t['Detection Times'].apply(np.min)
    det_t['Mean'] = det_t['Detection Times'].apply(np.mean)
    det_t['Median'] = det_t['Detection Times'].apply(np.median)
    det_t['Max'] = det_t['Detection Times'].apply(np.max)
    det_t['Count'] = det_t['Detection Times'].apply(len)
    
    del det_t['Detection Times']
    
    return det_t


def detection_time_to_impact(detection_time, impact_data):
    """
    Coverts detection time to an impact/damage metric.
    
    The impact DataFrame returned from this function can be used as input to
    :py:class:`ImpactFormulation<chama.optimization.ImpactFormulation>`.
    
    Parameters
    ----------
    detection_time : pandas DataFrame
        Detection time for each scenario-sensor pair. The DataFrame has 
        columns 'Scenario', 'Sensor', and 'T'. Note the 'T' column here is a
        single time and not a list of detection times.
    impact_data : pandas DataFrame
        Impact data for each scenario and time. The DataFrame has
        columns 'T' and one column for each scenario (user defined names)
        containing the impact/damage if each scenario was first detected at
        the times in 'T'.
        
    Returns
    ----------
    det_damage : pandas DataFrame
        DataFrame with columns 'Scenario', 'Sensor', and 'Impact'.
    """
    impact_data = impact_data.set_index('T')
    allT = list(set(detection_time['T']) | set(impact_data.index))
    impact_data = impact_data.reindex(allT)
    impact_data.sort_index(inplace=True)
    impact_data.interpolate(inplace=True)
    
    det_damage = detection_time.copy()
    det_damage['T'] = impact_data.lookup(detection_time['T'],
                                         detection_time['Scenario'])
    det_damage.rename(columns={'T': 'Impact'}, inplace=True)

    return det_damage


def detection_times_to_coverage(detection_times, coverage_type='scenario',
                                scenario=None):
    """
    Converts a detection times DataFrame to a coverage DataFrame

    The returned coverage DataFrame can be used for input to a
    :py:class:`CoverageFormulation<chama.optimization.CoverageFormulation>`.

    Parameters
    ----------
    detection_times : pandas DataFrame
        Detection times for each scenario-sensor pair. The DataFrame has
        columns  'Scenario', 'Sensor', and 'Detection Times', see
        :class:`~chama.impact.detection_times`.
    coverage_type : 'scenario' or 'scenario-time'
        Sets the coverage type: 'scenario' (the default value) builds lists
        of which scenarios are detected/covered by each sensor ignoring
        the time it was detected, 'scenario-time' treats every scenario-time
        pair as a new scenario and builds lists of which of these new
        scenarios are detected/covered by each sensor thereby calculating
        coverage over all scenarios and times.
    scenario : pandas DataFrame
        This is an optional argument which should be provided only if the
        coverage_type is 'scenario-time' and the user wants to
        propagate a scenario's undetected impact and probability to the new
        'scenario-time' scenarios. This DataFrame contains three columns,
        'Scenario` is the name of the scenarios, 'Undetected Impact' is the
        impact if the scenario goes undetected and 'Probability' is the
        probability or weighting of each scenario.

    Returns
    -------
    coverage : pandas DataFrame
        DataFrame with columns 'Sensor' and 'Coverage' where the 'Coverage'
        column contains a list of the scenarios/entities detected by a sensor.

    new_scenario : pandas DataFrame
        DataFrame returned if coverage_type is 'scenario-time' and a
        'scenario' DataFrame was provided. The columns in this DataFrame
        match those in the provided 'scenario' DataFrame

    """
    # remove any entries where detection times is an empty list
    detection_times = detection_times[detection_times.apply(
        lambda x: len(x['Detection Times']) != 0, axis=1)]
    if coverage_type == 'scenario':
        coverage = detection_times
        # drop the detection times
        coverage.drop('Detection Times', axis=1, inplace=True)
        coverage = coverage.groupby('Sensor')['Scenario'].unique()
        coverage = coverage.reset_index()
        coverage.rename(columns={'Scenario': 'Coverage'}, inplace=True)
        return coverage
    elif coverage_type == 'scenario-time':
        # create a series that has the Detection Times as the main data
        det_series = detection_times.set_index(
            ['Scenario', 'Sensor'])['Detection Times']

        # turn the Detection times list into a series. This creates a new
        # DataFrame with additional columns equal to the maximum number of
        # detection times with NaNs
        df = det_series.apply(pd.Series)

        # turn the additional columns into additional rows - This will add an
        # additional index to the multi-index whose value is the original
        # column number we also set the names appropriately
        df.columns.name = 'Detection Time Idx'
        df = df.stack()
        df.name = 'Detection Time'

        # make all the indices columns again
        df = df.reset_index()

        # add in the probabilities column if scenario DataFrame is provided
        if scenario is not None:
            df = pd.merge(df, scenario, how='left', on='Scenario')

        # rename the scenarios
        def rename_with_detection_times(row, col_name):
            value = row['Detection Time']
            return '{0}-{1}'.format(row[col_name], value)

        df['Scenario'] = df.apply(rename_with_detection_times,
                                  col_name='Scenario', axis=1)

        # drop the unnecessary columns
        df.drop(['Detection Time Idx', 'Detection Time'], inplace=True, axis=1)
        new_scenario = df.copy()
        new_scenario.drop(['Sensor'], inplace=True, axis=1)
        new_scenario.drop_duplicates(inplace=True)

        # group all the scenarios for each sensor into a list
        coverage = \
            df[['Scenario', 'Sensor']].groupby('Sensor')['Scenario'].unique()
        coverage = coverage.reset_index()

        # rename the columns for coverage
        coverage.rename(columns={'Scenario': 'Coverage'}, inplace=True)

        if scenario is None:
            # Only return coverage if no scenario DataFrame was provided
            return coverage

        return coverage, new_scenario

    raise ValueError("coverage_type must be 'scenario' or 'scenario-time'")


def impact_to_coverage(impact, impact_col_name='Impact'):
    """
    Convert an impact DataFrame to a coverage DataFrame

    The returned coverage DataFrame can be used for input to a
    :py:class:`CoverageFormulation<chama.optimization.CoverageFormulation>`.

    Parameters
    ----------
    impact : pandas DataFrame
        DataFrame containing three columns. 'Scenario' is the name of the 
        scenarios, 'Sensor' is the name of the sensors, and a third column 
        (called impact_col_name) contains an impact value.
    impact_col_name : str
        The name of the column containing the impact data (default = 'Impact')

    Returns
    -------
    coverage : pandas DataFrame
        DataFrame with columns 'Sensor' and 'Coverage' to be used as input
        to a coverage-based sensor placement solver.

    """
    coverage = impact.copy()
    # drop the impact column
    coverage.drop(impact_col_name, axis=1, inplace=True)
    coverage = coverage.groupby('Sensor')['Scenario'].unique()
    coverage = coverage.reset_index()
    coverage.rename(columns={'Scenario': 'Coverage'}, inplace=True)

    return coverage
