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
  used to simulate gas concentrations. Multiple scenarios capture
  uncertainty in the leak rate, leak location, wind speed and direction.
  Depending on the region of interest and the complexity of the system, very
  detailed or simple models can be used. In this case, the **signal** is
  concentration.

* **To place sensors to detect contaminant in a water distribution system**, 
  a water network model can be used to simulate hydraulics and water quality. 
  Multiple scenarios capture uncertainty in the location, rate, start time, 
  and duration of the injection along with uncertainty in customer demands. 
  EPANET [Ross00]_, WNTR [KHMB17]_, or similar water network simulators, can be 
  used to run this type of analysis. In this case, the **signal** is 
  concentration. 
  
* **To place sensors to detect a seismic event**, a wave propagation model can
  be used to simulate displacement. Multiple scenarios capture uncertainty
  in the location and magnitude of the seismic event along with subsurface
  heterogeneity. Depending on the region of interest and the complexity of
  the system, very detailed or simple models can be used. In this case, the
  **signal** is displacement.
  
For each scenario, the time, location, and signal are recorded. 
The points used to record time and location can be sparse to help reduce
data size. Chama uses Pandas DataFrames [Mcki13]_ to store the signal data. Pandas
includes many functions to easily populate DataFrames from a wide range of
file formats. For example, DataFrames can be generated from Excel, CSV, and
SQL files. 
Signal data can be stored in XYZ or Node format, as described below.

XYZ format
------------
In XYZ format, the X, Y, and Z location is stored for each entry.
In the DataFrame, X, Y, and Z describe the location, T is the simulation time, and Sn is
the signal for scenario n.  Exact column names must be used for X, Y, Z, and T. 
The scenario names can be defined by the user.
When using this format, Chama can interpolate sensor
measurements that are not represented in the signal data.
An example signal DataFrame in XYZ format is shown below using a simple 
2x2x2 system with three time steps and fabricated data for three scenarios.

.. doctest::
    :hide:

    >>> import numpy as np
    >>> import pandas as pd
    >>> x, y, z, t = np.meshgrid([1, 2], [1, 2], [1, 2], [0, 10, 20])
    >>> signal = pd.DataFrame({'X': z.flatten(),'Y': x.flatten(),'Z': y.flatten(),'T': t.flatten(),
    ...		'S1': [0,0,0,0.25,0.32,0.45,0.23,0.64,0.25,0.44,0.25,0.82,0.96,0.61,0.92,0.41,0.42,0,0,0,0,0,0,0],
    ...		'S2': [0,0,0,0.21,0.14,0.58,0.47,0.12,0.54,0.15,0.28,0.12,0.53,0.23,0.82,0.84,0.87,0.51,0,0,0,0,0,0],
    ...       'S3': [0,0.01,0,0.2,0.25,0.61,0.32,0.15,0.24,0.45,0.68,0.13,0.64,0.21,0.92,0.75,0.98,0.55,0.13,0,0,0,0,0]})
    >>> signal = signal[['X', 'Y', 'Z', 'T', 'S1','S2', 'S3']]

.. doctest::

    >>> print(signal)
        X  Y  Z   T    S1    S2    S3
    0   1  1  1   0  0.00  0.00  0.00
    1   1  1  1  10  0.00  0.00  0.01
    2   1  1  1  20  0.00  0.00  0.00
    3   2  1  1   0  0.25  0.21  0.20
    4   2  1  1  10  0.32  0.14  0.25
    5   2  1  1  20  0.45  0.58  0.61
    6   1  2  1   0  0.23  0.47  0.32
    7   1  2  1  10  0.64  0.12  0.15
    8   1  2  1  20  0.25  0.54  0.24
    9   2  2  1   0  0.44  0.15  0.45
    10  2  2  1  10  0.25  0.28  0.68
    11  2  2  1  20  0.82  0.12  0.13
    12  1  1  2   0  0.96  0.53  0.64
    13  1  1  2  10  0.61  0.23  0.21
    14  1  1  2  20  0.92  0.82  0.92
    15  2  1  2   0  0.41  0.84  0.75
    16  2  1  2  10  0.42  0.87  0.98
    17  2  1  2  20  0.00  0.51  0.55
    18  1  2  2   0  0.00  0.00  0.13
    19  1  2  2  10  0.00  0.00  0.00
    20  1  2  2  20  0.00  0.00  0.00
    21  2  2  2   0  0.00  0.00  0.00
    22  2  2  2  10  0.00  0.00  0.00
    23  2  2  2  20  0.00  0.00  0.00

Node format
--------------
In Node format, a location index is stored for each entry.  The index can 
be a string, integer, or float.
This format is useful when working with sparse systems, such as nodes in a networks.
In the DataFrame, Node is the location index, T is the simulation time, and Sn is
the signal for scenario n.  Exact column names must be used for Node and T. 
The scenario names can be defined by the user.
When using this format, Chama does not interpolate sensor
measurements and only stationary point sensors can be used to extract detection time.
An example signal DataFrame in Node format is shown below using 4 nodes
with three time steps and fabricated data for three scenarios.

.. doctest::
    :hide:

    >>> j, t = np.meshgrid([1, 2, 3, 4], [0, 10, 20])
    >>> signal = pd.DataFrame({'Node': j.flatten(), 'T': t.flatten(),
    ...		'S1': [0,0,0,0.25,0.32,0.45,0.23,0.64,0.25,0.44,0.25,0.82],
    ...		'S2': [0,0,0,0.21,0.14,0.58,0.47,0.12,0.54,0.15,0.28,0.12],
    ...		'S3': [0,0.01,0,0.2,0.25,0.61,0.32,0.15,0.24,0.45,0.68,0.13]})
    >>> signal = signal[['Node', 'T', 'S1','S2', 'S3']]
    >>> signal['Node'] =['n'+str(j) for j in signal['Node']]
    >>> signal = signal.sort_values('Node')
    >>> signal.reset_index(drop=True, inplace=True)

.. doctest::

    >>> print(signal)
       Node   T    S1    S2    S3
    0    n1   0  0.00  0.00  0.00
    1    n1  10  0.32  0.14  0.25
    2    n1  20  0.25  0.54  0.24
    3    n2   0  0.00  0.00  0.01
    4    n2  10  0.45  0.58  0.61
    5    n2  20  0.44  0.15  0.45
    6    n3   0  0.00  0.00  0.00
    7    n3  10  0.23  0.47  0.32
    8    n3  20  0.25  0.28  0.68
    9    n4   0  0.25  0.21  0.20
    10   n4  10  0.64  0.12  0.15
    11   n4  20  0.82  0.12  0.13
	
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

Initialize the Gaussian plume model and run (the first 5 rows of the signal
 DataFrame are printed):

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
simulation engines, for example, atmospheric dispersion can be simulated using 
AERMOD [USEPA04]_ or CALPUFF [ScSY00]_ or using detailed CFD models, transport 
in pipe networks can be simulated using EPANET [Ross00]_ or WNTR [KHMB17]_, and 
groundwater transport can be simulated using MODFLOW [McHa88]_. Output from 
external simulation engines can be easily formatted and imported into Chama.
