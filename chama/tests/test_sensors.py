from nose.tools import *
from os.path import abspath, dirname, join
import chama
import pandas as pd
import numpy as np

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')


def test_stationary_point_sensor():

    pos = chama.sensors.Stationary(location=(1, 2, 3))
    det = chama.sensors.Point(sample_times=[4, 5, 6], threshold=1E-3)
    sensor = chama.sensors.Sensor(position=pos, detector=det)

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
    det = chama.sensors.Point(sample_times=[1, 2, 3, 4, 5], threshold=1E-3)
    sensor = chama.sensors.Sensor(position=pos, detector=det)
    
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


def test_stationary_camera_sensor():
    t = 0.0
    X = np.linspace(-200, 200, 41)
    Y = np.linspace(-200, 200, 41)
    Z = np.linspace(0, 10, 11)

    conc = 10E-3

    allpoints = [[t, x0, y0, z0, conc] for x0 in X for y0 in Y for z0 in Z]
    signal = pd.DataFrame.from_records(allpoints,
                                       columns=['T', 'X', 'Y', 'Z', 'S1'])

    signal = signal.set_index(['T', 'X', 'Y', 'Z'])

    camloc = (0, 0, 0)
    camdir = (1, 1, 1)

    detector = chama.sensors.Camera(threshold=400, sample_times=[t],
                                    direction=camdir)
    pos = chama.sensors.Stationary(location=camloc)
    sensor = chama.sensors.Sensor(position=pos, detector=detector)

    assert_equal(sensor.detector.threshold, 400)
    assert_list_equal(sensor.detector.sample_times, [t])

    detected = sensor.get_detected_signal(signal)

    assert_equal(list(detected.values)[0], 76800)
