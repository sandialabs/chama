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

* **To place sensors to detect a gas leak**, an air dispersion model can be used
  to simulate gas concentrations. Multiple scenarios can capture the
  uncertainty in the leak rate, leak location, wind speed and direction.
  Depending on the region of interest and the complexity of the system, very
  detailed or coarse models can be used. In this case, the **signal** is
  concentration.

* **To place sensors to detect a seismic event**, a wave propagation model can
  be used to simulate displacement. Multiple scenarios can capture uncertainty
  in the location and magnitude of the seismic event along with subsurface
  heterogeneity. Depending on the region of interest and the complexity of
  the system, very detailed or coarse models can be used. In this case, the
  **signal** is displacement.

* **To place sensors to detect contaminant in a water distribution system**, 
  a water network model can be used to simulate hydraulics and water quality 
  given a set of contamination incidents. Multiple scenarios can capture 
  uncertainty in the location, injection rate, start time, and duration of 
  the incident along with changes in hydraulics due to different water demands. 
  EPANET, or similar water network models, are typically used to run this 
  type of analysis. In this case, the **signal** is concentration. 
  
For each scenario, the time, location (x,y,z), and signal are recorded. The
grid used to record time and location can be sparse. This can help reduce
the size of the data. Chama includes methods to interpolate sensor
measurements that are not represented in the signal data.

Chama uses Pandas DataFrames [Mcki13]_to store the signal data. Pandas
includes many functions to easily populate DataFrames from a wide range of
file formats. For example, DataFrames can be generated from Excel, CSV, and
SQL. The format of a signal DataFrame is shown in :numref:`fig-signal-format`,
where X, Y, and Z describe the location, T is the simulation time, and Sn is
the signal for scenario n.

.. _fig-signal-format:
.. figure:: figures/signalformat.png
   :scale: 50 %
   :alt: Chama flowchart
   
   Signal DataFrame format.

Internal simulation engines
---------------------------
Chama includes methods to run simple Gaussian plume and Gaussian puff air
dispersion models [Arya99]_. Both models assume that air dispersion follows a Gaussian
distribution. Gaussian plume models are used to model continuous sources,
while Gaussian puff models are used to model non-continuous or variable
sources. The :mod:`chama.transport` module has additional information on
running the Gaussian plume and Gaussian puff models.
Note that many air dispersion applications will require more sophisticated 
transport models.

The following simple example runs a single Gaussian plume model for a given receptor grid,
source, and atmospheric conditions.  To get started, import the required Python packages:

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

Initialize the Gaussian plume model and run (the first 5 rows of the signal DataFrame is printed):

.. doctest::

    >>> gauss_plume = chama.transport.GaussianPlume(grid, source, atm)
    >>> gauss_plume.run()
    >>> signal = gauss_plume.conc
    >>> chama.graphics.signal_convexhull(signal, 'S', 0.001)

    
External simulation engines
---------------------------
The transport simulations can be generated from a wide range of external
simulation engines, for example, air dispersion can be simulated using AERMOD
[USEPA04]_ or CALPUFF [ScSY00]_, transport in pipe networks can be simulated
using EPANET [Ross00]_, and groundwater transport can be simulated using
MODFLOW [McHa88]_. Output from external simulation engines can be easily
formatted and imported into Chama.
