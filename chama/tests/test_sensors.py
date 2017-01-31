from nose.tools import *
import numpy.testing as npt
from os.path import abspath, dirname, join
import chama
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

testdir = dirname(abspath(__file__))
datadir = join(testdir,'data')

def test_stationary_sensor():
    sensor = chama.sensors.Stationary()
    sensor.location = [1,2,3]
    sensor.sampling_times = [4,5,6]
    
    sample_points = sensor.get_sample_points()
    
    expected_sampling_points = [(4, 1, 2, 3), 
                                (5, 1, 2, 3),
                                (6, 1, 2, 3)]
    npt.assert_almost_equal(np.array(sample_points),
                            np.array(expected_sampling_points), decimal=4)

def test_mobile_sensor_base():
    sensor = chama.sensors.Mobile()
    sensor.locations = [[0,0,0], [1,0,0], [1,3,0], [1,2,1]]
    sensor.speed = 1
    sensor.camera = None
    sensor.sampling_times = [1,2,3,4,5]
    sensor.repeat = False
    
    sample_points = sensor.get_sample_points()
    
    expected_sampling_points = [(1, 1.0, 0.0, 0.0), 
                                (2.0, 1.0, 1.0, 0.0),
                                (3.0, 1.0, 2.0, 0.0), 
                                (4.0, 1.0, 3.0, 0.0), 
                                (5.0, 1.0, 3-np.sqrt(0.5), np.sqrt(0.5))]
    npt.assert_almost_equal(np.array(sample_points),
                            np.array(expected_sampling_points), decimal=4)

#    pathx = []
#    pathy = []
#    pathz = []
#    for i in range(len(path_sensor.locations)):
#        pathx.append(path_sensor.locations[i][0])
#        pathy.append(path_sensor.locations[i][1])
#        pathz.append(path_sensor.locations[i][2])
#        
#    samplex = []
#    sampley = []
#    samplez = []
#    for i in range(len(sample_points)):
#        samplex.append(sample_points[i][1])
#        sampley.append(sample_points[i][2])
#        samplez.append(sample_points[i][3])
#
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    ax.plot(pathx, pathy, pathz)
#    plt.hold(True)
#    ax.scatter(samplex,sampley, samplez, s=40)
#    ax.set_aspect('equal', 'datalim')

def test_mobile_sensor_time_limited():
    sensor = chama.sensors.Mobile()
    sensor.locations = [[0,0,0], [1,0,0], [1,3,0], [1,2,1]]
    sensor.speed = 1
    sensor.camera = None
    sensor.sampling_times = [1,2]
    sensor.repeat = False
    
    sample_points = sensor.get_sample_points()
    
    expected_sampling_points = [(1, 1.0, 0.0, 0.0), 
                                (2.0, 1.0, 1.0, 0.0)]
    npt.assert_almost_equal(np.array(sample_points),
                            np.array(expected_sampling_points), decimal=4)
                            
def test_mobile_sensor_extra_time():
    sensor = chama.sensors.Mobile()
    sensor.locations = [[0,0,0], [1,0,0], [1,3,0], [1,2,1]]
    sensor.speed = 1
    sensor.camera = None
    sensor.sampling_times = [1,2,6]
    sensor.repeat = False
    
    sample_points = sensor.get_sample_points()
    
    expected_sampling_points = [(1, 1.0, 0.0, 0.0), 
                                (2.0, 1.0, 1.0, 0.0)]
    npt.assert_almost_equal(np.array(sample_points),
                            np.array(expected_sampling_points), decimal=4)

#def test_path_sensor_extra_time_repeat_path():
#    path_sensor = chama.sensors.Path()
#    path_sensor.locations = [[0,0,0], [1,0,0], [1,3,0], [1,2,1]]
#    path_sensor.speed = 1
#    path_sensor.camera = None
#    path_sensor.sampling_times = [1,2,3,4,5,6]
#    path_sensor.repeat = True
#    
#    sample_points = path_sensor.get_sample_points()
#    
#    expected_sampling_points = [(1, 1.0, 0.0, 0.0), 
#                                (2.0, 1.0, 1.0, 0.0),
#                                (3.0, 1.0, 2.0, 0.0), 
#                                (4.0, 1.0, 3.0, 0.0), 
#                                (5.0, 1.0, 3-np.sqrt(0.5), np.sqrt(0.5)),
#                                (6.0, np.nan, np.nan, np.nan)] # CALCULATE this point
#    npt.assert_almost_equal(np.array(sample_points),
#                            np.array(expected_sampling_points), decimal=4)
                            
if __name__ == '__main__':
    test_mobile_sensor_base()