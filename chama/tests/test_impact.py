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
                                columns=['Scenario', 'Sensor', 'Detection Times'])
        
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
        self.signal = pd.DataFrame({'Node': j.flatten(),
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
                                columns=['Scenario', 'Sensor', 'Detection Times'])
        
        impact.set_index('Sensor', inplace=True)
        expected.set_index('Sensor', inplace=True)
        assert_frame_equal(impact, expected, check_dtype=False,
                           check_like=True)

    def test_extract2(self):
        self.signal['Node'] =['n'+str(j) for j in self.signal['Node']]
         
        sensors = {}
        pos1 = chama.sensors.Stationary(location='n1')
        det1 = chama.sensors.Point(sample_times=[0], threshold=0)
        sensors['A'] = chama.sensors.Sensor(position=pos1, detector=det1)

        pos2 = chama.sensors.Stationary(location='n2')
        det2 = chama.sensors.Point(sample_times=[0, 10], threshold=2)
        sensors['B'] = chama.sensors.Sensor(position=pos2, detector=det2)

        pos3 = chama.sensors.Stationary(location='n3')
        det3 = chama.sensors.Point(sample_times=[0, 10, 20], threshold=40)
        sensors['C'] = chama.sensors.Sensor(position=pos3, detector=det3)
        
        impact = chama.impact.detection_times(self.signal, sensors)

        expected = pd.DataFrame([('S', 'A', [0]),
                                 ('S', 'B', [10]),
                                 ('S', 'C', [10,20])],
                                columns=['Scenario', 'Sensor', 'Detection Times'])
        
        impact.set_index('Sensor', inplace=True)
        expected.set_index('Sensor', inplace=True)
        assert_frame_equal(impact, expected, check_dtype=False,
                           check_like=True)

class TestCoverageConversions(unittest.TestCase):
    def test_impact_to_coverage(self):
        pass

    def test_detection_times_to_coverage(self):
        scenario = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Undetected Impact': [48.0, 250.0, 100.0],
            'Probability': [0.1, 0.1, 0.8]})
        detection_times = pd.DataFrame({
            'Scenario': ['S1', 'S2', 'S3'],
            'Sensor': ['A', 'A', 'B'],
            'Detection Times': [[2, 3, 4], [3], [4, 5]]})

        coverage1 = chama.impact.detection_times_to_coverage(detection_times=detection_times,
                                                                        coverage_type='scenario')
        coverage1_expected = pd.DataFrame({'Sensor': ['A', 'B'],
                                           'Coverage': [['S1', 'S2'], ['S3']]
                                           })
        assert_frame_equal(coverage1.set_index('Sensor'), coverage1_expected.set_index('Sensor'))

        coverage2, scenario2 = chama.impact.detection_times_to_coverage(detection_times=detection_times,
                                                                        coverage_type='scenario-time')
        coverage2_expected = pd.DataFrame({'Sensor': ['A', 'B'],
                                           'Coverage': [['S1-2.0', 'S1-3.0', 'S1-4.0', 'S2-3.0'], ['S3-4.0', 'S3-5.0']]})
        scenario2_expected = pd.DataFrame({'Scenario': ['S1-2.0', 'S1-3.0', 'S1-4.0', 'S2-3.0', 'S3-4.0', 'S3-5.0']})
        assert_frame_equal(coverage2.set_index('Sensor'), coverage2_expected.set_index('Sensor'))
        assert_frame_equal(scenario2.set_index('Scenario'), scenario2_expected.set_index('Scenario'))

        coverage3, scenario3 = chama.impact.detection_times_to_coverage(detection_times=detection_times,
                                                                        coverage_type='scenario-time',
                                                                        scenario=scenario)
        scenario3_expected = pd.DataFrame({'Scenario': ['S1-2.0', 'S1-3.0', 'S1-4.0', 'S2-3.0', 'S3-4.0', 'S3-5.0'],
                                           'Undetected Impact': [48.0, 48.0, 48.0, 250.0, 100.0, 100.0],
                                           'Probability': [0.1, 0.1, 0.1, 0.1, 0.8, 0.8]})
        assert_frame_equal(coverage3.set_index('Sensor'), coverage2_expected.set_index('Sensor'))
        assert_frame_equal(scenario3.set_index('Scenario'), scenario3_expected.set_index('Scenario'))
