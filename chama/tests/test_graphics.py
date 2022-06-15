import unittest
import os
from os.path import abspath, dirname, join, isfile
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import chama

testdir = dirname(abspath(__file__))


class TestSignalGraphics(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        x_grid = np.linspace(-100, 100, 21)
        y_grid = np.linspace(-100, 100, 21)
        z_grid = np.linspace(0, 40, 21)
        self.grid = chama.simulation.Grid(x_grid, y_grid, z_grid)

        self.source = chama.simulation.Source(-20, 20, 1, 1.5)

        self.atm = pd.DataFrame({'Wind Direction': [45, 120, 200],
                                 'Wind Speed': [1.2, 1, 1.8],
                                 'Stability Class': ['A', 'B', 'C']},
                                index=[0, 10, 20])
    
        gauss_plume = chama.simulation.GaussianPlume(self.grid, self.source, 
                                                     self.atm)
        self.signal = gauss_plume.conc
        
    @classmethod
    def tearDownClass(self):
        pass

    def test_signal_convexhull(self):
        filename = abspath(join(testdir, 'plot_signal_convexhull1.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        chama.graphics.signal_convexhull(self.signal, ['S'], 0.001)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))
    
    def test_signal_xsection(self):
        filename = abspath(join(testdir, 'plot_signal_xsection1.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        chama.graphics.signal_xsection(self.signal, 'S')
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))

    def test_signal_animate(self):
        pass


class TestSensorGraphics(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        
        sensors = {}
        pos1 = chama.sensors.Stationary(location=(1, 2, 3))
        det1 = chama.sensors.Point(threshold=0.001,
                                   sample_times=[0, 2, 4, 6, 8, 10])
        sensorA = chama.sensors.Sensor(position=pos1, detector=det1)
        sensors['A'] = sensorA
       
        pos2 = chama.sensors.Mobile(locations=[(0, 1, 1), (1, 2, 2),
                                               (1, 3, 0), (1, 2, 1)],
                                    speed=0.5)
        det2 = chama.sensors.Camera(threshold=100, sample_times=[0, 3, 6, 9],
                                    direction=(1, 1, 1))
        sensorB = chama.sensors.Sensor(position=pos2, detector=det2)
        sensors['B'] = sensorB
              
        self.sensors = sensors
    
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_sensor_locations(self):
        filename = abspath(join(testdir, 'plot_sensors.png'))
        if isfile(filename):
            os.remove(filename)
        
        plt.figure()
        chama.graphics.sensor_locations(self.sensors)
        plt.savefig(filename, format='png')
        plt.close()
        
        self.assertTrue(isfile(filename))

if __name__ == "__main__":
    unittest.main()
