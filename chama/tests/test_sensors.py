import unittest
import pandas as pd
import numpy as np
import chama


class TestSensors(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_stationary_point_sensor(self):
    
        pos = chama.sensors.Stationary(location=(1, 2, 3))
        det = chama.sensors.Point(sample_times=[4, 5, 6], threshold=1E-3)
        sensor = chama.sensors.Sensor(position=pos, detector=det)
    
        self.assertTupleEqual(sensor.position.location, (1, 2, 3))
        self.assertListEqual(sensor.detector.sample_times, [4, 5, 6])
        self.assertEqual(sensor.detector.threshold, 1E-3)
    
        sample_points = sensor.detector.get_sample_points(sensor.position)
        
        expected_sampling_points = [(4, 1, 2, 3), 
                                    (5, 1, 2, 3),
                                    (6, 1, 2, 3)]
    
        for i in expected_sampling_points:
            self.assertIn(i, sample_points)
    
    
    def test_mobile_point_sensor(self):
    
        pos = chama.sensors.Mobile(locations=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 3, 0),
                (1, 2, 1)],
                speed=1, start_time=0, repeat=False)
        det = chama.sensors.Point(sample_times=[0, 1, 2, 3, 4, 5, 6, 7, 8], 
                                  threshold=1E-3)
        sensor = chama.sensors.Sensor(position=pos, detector=det)
        
        sample_points = sensor.detector.get_sample_points(sensor.position)
        expected_sampling_points = [
                (0, 0, 0, 0),
                (1, 1, 0, 0), 
                (2, 1, 1, 0),
                (3, 1, 2, 0), 
                (4, 1, 3, 0), 
                (5, 1, round(3 - np.sqrt(0.5), 5), round(np.sqrt(0.5), 5)),
                (6, 1, 2, 1),
                (7, 1, 2, 1),
                (8, 1, 2, 1)]
        np.testing.assert_almost_equal(sample_points, expected_sampling_points, 5)
        
        # Change start time
        pos.start_time = 2
        det.sample_points = None
        sample_points2 = sensor.detector.get_sample_points(sensor.position)
        expected_sampling_points2 = [
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (2, 0, 0, 0),
                (3, 1, 0, 0), 
                (4, 1, 1, 0),
                (5, 1, 2, 0), 
                (6, 1, 3, 0), 
                (7, 1, round(3 - np.sqrt(0.5), 5), round(np.sqrt(0.5), 5)),
                (8, 1, 2, 1)]
        np.testing.assert_almost_equal(sample_points2,
                                       expected_sampling_points2, 5)
        
        # Repeat path
        pos.start_time = 0
        pos.repeat = True
        det.sample_points = None
        sample_points3 = sensor.detector.get_sample_points(sensor.position)
        temp = 1 - np.sqrt(pow(1 - pos(5)[0], 2) + pow(2 - pos(5)[1], 2) +
                           pow(1 - pos(5)[2], 2))
        expected_sampling_points = [
                (0, 0, 0, 0),
                (1, 1, 0, 0), 
                (2, 1, 1, 0),
                (3, 1, 2, 0), 
                (4, 1, 3, 0), 
                (5, 1, round(3 - np.sqrt(0.5), 5), round(np.sqrt(0.5), 5)),
                (6, temp, 0, 0),
                (7, 1, temp, 0),
                (8, 1, 1+temp, 0)]
        np.testing.assert_almost_equal(sample_points3, expected_sampling_points, 5)
    
    
    def test_stationary_camera_sensor(self):
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
    
        self.assertEqual(sensor.detector.threshold, 400)
        self.assertListEqual(sensor.detector.sample_times, [t])
    
        detected = sensor.get_detected_signal(signal)
    
        self.assertEqual(list(detected.values)[0], 76800)

if __name__ == "__main__":
    unittest.main()
