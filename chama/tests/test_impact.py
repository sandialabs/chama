import unittest
from os.path import abspath, dirname, join
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
import chama
        
testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')

class TestImpact(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        sensors = {}
        sensors['sensor1'] = chama.sensors.Sensor(sample_times=[0], location=(1,2,3),threshold=0)
        sensors['sensor2'] = chama.sensors.Sensor(sample_times=[0,10], location=(2,2,2),threshold=2)
        sensors['sensor3'] = chama.sensors.Sensor(sample_times=[0,10,20], location=(3,2,1),threshold=40)
        self.sensors = sensors
         
        x,y,z,t = np.meshgrid([1,2,3], [1,2,3], [1,2,3], [0,10,20])       
        self.signal = pd.DataFrame({'X': x.flatten(),
                               'Y': y.flatten(),
                               'Z': z.flatten(),
                               'T': t.flatten(),
                               'S': x.flatten()*t.flatten()})
        
    @classmethod
    def tearDownClass(self):
        pass

    def test_extract(self):
        impact = chama.impact.extract(self.signal, self.sensors)

        expected = pd.DataFrame([('S', 'sensor1', 0),
                                 ('S', 'sensor2', 10),
                                 ('S', 'sensor3', 20)],
                    columns=['Scenario', 'Sensor', 'Impact'])
        
        assert_frame_equal(impact.sort_values('Impact').reset_index(inplace=True), 
                           expected.sort_values('Impact').reset_index(inplace=True), 
                            check_dtype=False)
    
    #def test_extract_with_interpolation(self):
    #    new_sensors = self.sensors
    #    new_sensors['sensor4'] = chama.sensors.Sensor(sample_times=[0], location=(1.5,1.5,1.5),threshold=3)
    #    impact = chama.impact.extract(self.signal, new_sensors)
