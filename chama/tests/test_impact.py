import unittest
from os.path import abspath, dirname, join
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
import chama

testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')


class TestImpact_XYZ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        sensors = {}
        pos1 = chama.sensors.Stationary(location=(1, 2, 3))
        det1 = chama.sensors.Point(sample_times=[0], threshold=0)
        sensors['A'] = chama.sensors.Sensor(position=pos1, detector=det1)

        pos2 = chama.sensors.Stationary(location=(2, 2, 2))
        det2 = chama.sensors.Point(sample_times=[0, 10], threshold=2)
        sensors['B'] = chama.sensors.Sensor(position=pos2, detector=det2)

        pos3 = chama.sensors.Stationary(location=(3, 2, 1))
        det3 = chama.sensors.Point(sample_times=[0, 10, 20], threshold=40)
        sensors['C'] = chama.sensors.Sensor(position=pos3, detector=det3)
        self.sensors = sensors

        x, y, z, t = np.meshgrid([1, 2, 3], [1, 2, 3], [1, 2, 3], [0, 10, 20, 30])
        self.signal = pd.DataFrame({'X': x.flatten(),
                                    'Y': y.flatten(),
                                    'Z': z.flatten(),
                                    'T': t.flatten(),
                                    'S': t.flatten() * t.flatten()})

    @classmethod
    def tearDownClass(self):
        pass

    def test_extract(self):
        impact = chama.impact.detection_times(self.signal, self.sensors)

        expected = pd.DataFrame([('S', 'A', [0]),
                                 ('S', 'B', [10]),
                                 ('S', 'C', [10,20])],
                                columns=['Scenario', 'Sensor', 'Impact'])
        
        impact.set_index('Sensor', inplace=True)
        expected.set_index('Sensor', inplace=True)
        assert_frame_equal(impact, expected, check_dtype=False,
                           check_like=True)

        # def test_extract_with_interpolation(self):
        #    new_sensors = self.sensors
        #    new_sensors['sensor4'] = chama.sensors.Sensor(sample_times=[0], location=(1.5,1.5,1.5),threshold=3)
        #    impact = chama.impact.extract(self.signal, new_sensors)

class TestImpact_J(unittest.TestCase):
    @classmethod
    def setUpClass(self):
    
        j, t = np.meshgrid([1, 2, 3, 4], [0, 10, 20])
        self.signal = pd.DataFrame({'J': j.flatten(),
                                    'T': t.flatten(),
                                    'S': t.flatten() * t.flatten()})

    @classmethod
    def tearDownClass(self):
        pass

    def test_extract1(self):
        sensors = {}
        pos1 = chama.sensors.Stationary(location=1)
        det1 = chama.sensors.Point(sample_times=[0], threshold=0)
        sensors['A'] = chama.sensors.Sensor(position=pos1, detector=det1)

        pos2 = chama.sensors.Stationary(location=2)
        det2 = chama.sensors.Point(sample_times=[0, 10], threshold=2)
        sensors['B'] = chama.sensors.Sensor(position=pos2, detector=det2)

        pos3 = chama.sensors.Stationary(location=3)
        det3 = chama.sensors.Point(sample_times=[0, 10, 20], threshold=40)
        sensors['C'] = chama.sensors.Sensor(position=pos3, detector=det3)
        
        impact = chama.impact.detection_times(self.signal, sensors)

        expected = pd.DataFrame([('S', 'A', [0]),
                                 ('S', 'B', [10]),
                                 ('S', 'C', [10,20])],
                                columns=['Scenario', 'Sensor', 'Impact'])
        
        impact.set_index('Sensor', inplace=True)
        expected.set_index('Sensor', inplace=True)
        assert_frame_equal(impact, expected, check_dtype=False,
                           check_like=True)

    def test_extract2(self):
        self.signal['J'] =['Node'+str(j) for j in self.signal['J']]
         
        sensors = {}
        pos1 = chama.sensors.Stationary(location='Node1')
        det1 = chama.sensors.Point(sample_times=[0], threshold=0)
        sensors['A'] = chama.sensors.Sensor(position=pos1, detector=det1)

        pos2 = chama.sensors.Stationary(location='Node2')
        det2 = chama.sensors.Point(sample_times=[0, 10], threshold=2)
        sensors['B'] = chama.sensors.Sensor(position=pos2, detector=det2)

        pos3 = chama.sensors.Stationary(location='Node3')
        det3 = chama.sensors.Point(sample_times=[0, 10, 20], threshold=40)
        sensors['C'] = chama.sensors.Sensor(position=pos3, detector=det3)
        
        impact = chama.impact.detection_times(self.signal, sensors)

        expected = pd.DataFrame([('S', 'A', [0]),
                                 ('S', 'B', [10]),
                                 ('S', 'C', [10,20])],
                                columns=['Scenario', 'Sensor', 'Impact'])
        
        impact.set_index('Sensor', inplace=True)
        expected.set_index('Sensor', inplace=True)
        assert_frame_equal(impact, expected, check_dtype=False,
                           check_like=True)