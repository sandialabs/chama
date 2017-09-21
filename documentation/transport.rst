.. raw:: latex

    \newpage

.. _transport:

Transport simulation
====================

Chama requires a set of precomputed transport simulations to determine
optimal sensor placement. The type of transport simulation depends on the
application and scale of interest. Multiple scenarios should be generated to
capture uncertainty in the system. For each scenario, a **signal** is recorded.

For example:

* **To place sensors to detect a gas leak**, an atmospheric dispersion model can be 
  used to simulate gas concentrations. Multiple scenarios can capture the
  uncertainty in the leak rate, leak location, wind speed and direction.
  Depending on the region of interest and the complexity of the system, very
  detailed or coarse models can be used. In this case, the **signal** is
  concentration.

* **To place sensors to detect contaminant in a water distribution system**, 
  a water network model can be used to simulate hydraulics and water quality. 
  Multiple scenarios can capture uncertainty in the location, rate, start time, 
  and duration of the injection along with uncertainty in customer demands. 
  EPANET, or similar water network models, are typically used to run this 
  type of analysis. In this case, the **signal** is concentration. 
  
* **To place sensors to detect a seismic event**, a wave propagation model can
  be used to simulate displacement. Multiple scenarios can capture uncertainty
  in the location and magnitude of the seismic event along with subsurface
  heterogeneity. Depending on the region of interest and the complexity of
  the system, very detailed or coarse models can be used. In this case, the
  **signal** is displacement.
  
For each scenario, the time, location (x,y,z), and signal are recorded. 
The points used to record time and location can be sparse to help reduce
data size. Chama includes methods to interpolate sensor
measurements that are not represented in the signal data.

Chama uses Pandas DataFrames [Mcki13]_ to store the signal data. Pandas
includes many functions to easily populate DataFrames from a wide range of
file formats. For example, DataFrames can be generated from Excel, CSV, and
SQL files. An example signal DataFrame is shown below using a simple 
2x2x2 system with three time steps and fabricated data for three scenarios.
X, Y, and Z describe the location, T is the simulation time, and Sn is
the signal for scenario n.  Exact column names must be used for X, Y, Z, and T. 
The scenario names can be defined by the user.

.. doctest::
    :hide:

    >>> import numpy as np
    >>> import pandas as pd
    >>> x, y, z, t = np.meshgrid([1, 2], [1, 2], [1, 2], [0, 10, 20])
    >>> signal = pd.DataFrame({'X': z.flatten(),'Y': x.flatten(),'Z': y.flatten(),'T': t.flatten(),
    ...		'S1': [0,0,0,0.2,0.32,0.45,0.23,0.64,0.25,0.44,0.25,0.82,0.96,0.61,0.92,0.41,0.42,0,0,0,0,0,0,0],
    ...		'S2': [0,0,0,0.2,0.14,0.58,0.47,0.12,0.54,0.15,0.28,0.12,0.53,0.23,0.82,0.84,0.87,0.51,0,0,0,0,0,0],
    ...     'S3': [0,0.01,0,0.2,0.14,0.58,0.47,0.12,0.54,0.45,0.68,0.12,0.53,0.23,0.82,0.84,0.87,0.51,0.13,0,0,0,0,0]})
    >>> signal = signal[['X', 'Y', 'Z', 'T', 'S1','S2', 'S3']]

.. doctest::

    >>> print(signal)
        X  Y  Z   T    S1    S2    S3
    0   1  1  1   0  0.00  0.00  0.00
    1   1  1  1  10  0.00  0.00  0.01
    2   1  1  1  20  0.00  0.00  0.00
    3   2  1  1   0  0.20  0.20  0.20
    4   2  1  1  10  0.32  0.14  0.14
    5   2  1  1  20  0.45  0.58  0.58
    6   1  2  1   0  0.23  0.47  0.47
    7   1  2  1  10  0.64  0.12  0.12
    8   1  2  1  20  0.25  0.54  0.54
    9   2  2  1   0  0.44  0.15  0.45
    10  2  2  1  10  0.25  0.28  0.68
    11  2  2  1  20  0.82  0.12  0.12
    12  1  1  2   0  0.96  0.53  0.53
    13  1  1  2  10  0.61  0.23  0.23
    14  1  1  2  20  0.92  0.82  0.82
    15  2  1  2   0  0.41  0.84  0.84
    16  2  1  2  10  0.42  0.87  0.87
    17  2  1  2  20  0.00  0.51  0.51
    18  1  2  2   0  0.00  0.00  0.13
    19  1  2  2  10  0.00  0.00  0.00
    20  1  2  2  20  0.00  0.00  0.00
    21  2  2  2   0  0.00  0.00  0.00
    22  2  2  2  10  0.00  0.00  0.00
    23  2  2  2  20  0.00  0.00  0.00

Internal simulation engines
---------------------------
Chama includes methods to run simple Gaussian plume and Gaussian puff atmospheric
dispersion models [Arya99]_. Both models assume that atmospheric dispersion follows a Gaussian
distribution. Gaussian plume models are typically used to model steady state plumes,
while Gaussian puff models are used to model non-continuous sources. 
The :mod:`chama.transport` module has additional information on
running the Gaussian plume and Gaussian puff models.
Note that many atmospheric dispersion applications require more sophisticated models.

The following simple example runs a single Gaussian plume model for a given receptor grid,
source, and atmospheric conditions.  

Import the required Python packages:

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chama
	
Define the receptor grid:

.. doctest::

    >>> x_grid = np.linspace(-100, 100, 21)
    >>> y_grid = np.linspace(-100, 100, 21)
    >>> z_grid = np.linspace(0, 40, 21)
    >>> grid = chama.transport.Grid(x_grid, y_grid, z_grid)

Define the source:

.. doctest::

    >>> source = chama.transport.Source(-20, 20, 1, 1.5)

Define the atmospheric conditions:

.. doctest::

    >>> atm = pd.DataFrame({'Wind Direction': [45, 60], 
    ...                     'Wind Speed': [1.2, 1], 
    ...                     'Stability Class': ['A', 'A']}, index=[0, 10])

Initialize the Gaussian plume model and run (the first 5 rows of the signal DataFrame are printed):

.. doctest::

    >>> gauss_plume = chama.transport.GaussianPlume(grid, source, atm)
    >>> gauss_plume.run()
    >>> signal = gauss_plume.conc
    >>> print(signal.head(5))
           X      Y    Z  T    S
    0 -100.0 -100.0  0.0  0  0.0
    1 -100.0 -100.0  2.0  0  0.0
    2 -100.0 -100.0  4.0  0  0.0
    3 -100.0 -100.0  6.0  0  0.0
    4 -100.0 -100.0  8.0  0  0.0

The Gaussian Puff model is run in a similar manner.  
The time between puffs (tpuff) and time at the end of the simulation (tend) must be defined.

Initialize the Gaussian puff model and run:

.. doctest::

    >>> gauss_puff = chama.transport.GaussianPuff(grid, source, atm, tpuff=1, tend=10)
    >>> gauss_puff.run(grid, 10)
    >>> signal = gauss_puff.conc

	
External simulation engines
---------------------------
Transport simulations can also be generated from a wide range of external
simulation engines, for example, atmospheric dispersion can be simulated using AERMOD
[USEPA04]_ or CALPUFF [ScSY00]_, transport in pipe networks can be simulated
using EPANET [Ross00]_, and groundwater transport can be simulated using
MODFLOW [McHa88]_. Output from external simulation engines can be easily
formatted and imported into Chama.
