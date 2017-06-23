from nose.tools import *
from os.path import abspath, dirname, join
import chama
import pandas as pd
import numpy as np

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')


def test_stationary_point_sensor():
    sensor = chama.sensors.Sensor(location=(1, 2, 3),
                                  sample_times=[4, 5, 6],
                                  threshold=1E-3)

    assert_tuple_equal(sensor.position.location, (1, 2, 3))
    assert_list_equal(sensor.detector.sample_times, [4, 5, 6])
    assert_equal(sensor.detector.threshold, 1E-3)

    sample_points = sensor.detector.get_sample_points(sensor.position)
    
    expected_sampling_points = [(4, 1, 2, 3), 
                                (5, 1, 2, 3),
                                (6, 1, 2, 3)]

    for i in expected_sampling_points:
        assert_in(i, sample_points)


def test_mobile_point_sensor():

    pos = chama.sensors.Mobile(locations=[(0, 0, 0),
                                          (1, 0, 0),
                                          (1, 3, 0),
                                          (1, 2, 1)],
                               speed=1)
    sensor = chama.sensors.Sensor(position=pos,
                                  sample_times=[1, 2, 3, 4, 5],
                                  threshold=1E-3)
    
    temp = sensor.detector.get_sample_points(sensor.position)
    sample_points = []
    for pt in temp:
        if pt[0] == 5.0 and pt[1] == 1.0:
            pt = (pt[0], pt[1], round(pt[2], 5), round(pt[3], 5))
        sample_points.append(pt)

    expected_sampling_points = [(1, 1.0, 0.0, 0.0), 
                                (2.0, 1.0, 1.0, 0.0),
                                (3.0, 1.0, 2.0, 0.0), 
                                (4.0, 1.0, 3.0, 0.0), 
                                (5.0, 1.0, round(3 - np.sqrt(0.5), 5),
                                 round(np.sqrt(0.5), 5))]
    assert_list_equal(sample_points, expected_sampling_points)
