"""
The impact module contains methods to extract the impact of detecting transport 
simulations given a set of defined sensor techninologies.
"""
from __future__ import print_function
import pandas as pd
import time


def extract(signal, sensors, metric='Time'):
    """
    Extract the impact metric from a signal profile and sensors
    """
    print('Beginning signal extraction...')

    # Extracting a subset of the signal in the sensor module is fastest
    # using multiindex even though setting the index initially is slow
    txyz = ['T', 'X', 'Y', 'Z']
    # check if the signal is already in multiindex form
    if not isinstance(signal.index, pd.MultiIndex):
        t0 = time.time()
        signal = signal.set_index(txyz)
        print('Time to set the index: ', time.time() - t0, ' sec')

    temp_impact = {'Scenario': [], 'Sensor': [], 'Impact': []}

    sensor_time = 0

    print("    Extract/Integrate...")
    for (name, sensor) in sensors.items():  # loop over sensors

        # Get detected signal
        t0 = time.time()
        detected = sensor.get_detected_signal(signal)
        t1 = time.time()
        # print('time: ', t1-t0, ' s')
        sensor_time = sensor_time + (t1 - t0)

        # If the sensor detected something
        if detected.shape[0] > 0:
            if metric == 'Coverage':
                # Rework this to remove the for loop
                for row in range(detected.shape[0]):
                    col = str(detected.index[row])
                    val = 0
                    temp_impact['Scenario'].append(col)
                    temp_impact['Sensor'].append(name)
                    temp_impact['Impact'].append(val)

            elif metric == 'Time':
                for col in set(detected.index.get_level_values(1)):
                    temp = detected.loc[:, col]  # this returns a series
                    val = temp.index.min()
                    temp_impact['Scenario'].append(col)
                    temp_impact['Sensor'].append(name)
                    temp_impact['Impact'].append(val)

    impact = pd.DataFrame()
    impact['Scenario'] = temp_impact['Scenario']
    impact['Sensor'] = temp_impact['Sensor']
    impact['Impact'] = temp_impact['Impact']

    impact = impact.sort_values('Scenario')
    impact = impact.reset_index(drop=True)
    
    print(sensor_time, ' sec')

    return impact


# def _translate(impact, metric):
#     """
#     Translate the time of detection to a different metric
#     """
#     # Gather sample points
#     scenario = impact.index.get_level_values(0)
#     t = impact.tolist()
#     sample_points = list(zip(scenario,t))
#
#     # Interpolate metrics (if needed)
#     metric = interpolate(metric, sample_points)
#
#     # There is probably a better way to do this without a loop
#     data = {}
#     for i in range(impact.shape[0]):
#         data[(impact.index[i][0], impact.index[i][1])] = metric[
#             impact.index[i][0]][impact.iloc[i]]
#
#     # Reindex
#     impact_metric = pd.Series(data)
#     impact_metric.index.names = ['Scenario', 'Sensor']
#
#     return impact_metric

# def _add_nondetected_impact(impact_metric, scenario_list,
#                             nondetected_impact_value):
#
#     multiindex = pd.MultiIndex.from_tuples(list(zip(scenario_list,
#                                                 len(scenario_list) * [
#                                                     '_NotDetected'])),
#                                        names=impact_metric.index.names)
#     nondetected_impact = pd.DataFrame(index=multiindex,
#                                       data=nondetected_impact_value,
#                                       columns=impact_metric.columns)
#     impact_metric = impact_metric.append(nondetected_impact)
#
#     return impact_metric
