import unittest
from os.path import abspath, dirname, join
import pandas as pd
import numpy as np
import chama
        
testdir = dirname(abspath(__file__))
datadir = join(testdir, 'data')


class TestGaussianModels(unittest.TestCase):

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

    @classmethod
    def tearDownClass(self):
        pass

    def test_gaussian_plume(self):
        gauss_plume = chama.simulation.GaussianPlume(self.grid, self.source, 
                                                     self.atm)
        signal = gauss_plume.conc

        self.assertAlmostEqual(signal['S'].max(), 0.033378, 4)

    def test_gaussian_puff(self):
        gauss_puff = chama.simulation.GaussianPuff(self.grid, self.source, 
                                                   self.atm, tpuff=0.1,
                                                   tend=20)
        signal = gauss_puff.conc

if __name__ == "__main__":
    unittest.main()
