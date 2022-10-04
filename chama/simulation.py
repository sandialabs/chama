"""
The simulation module contains methods to run Gaussian air dispersion models. 
Chama can also integrate simulations from third party software for additional
sensor placement applications.

.. rubric:: Contents

.. autosummary::

    Grid
    Source
    GaussianPlume
    GaussianPuff
"""

from __future__ import print_function, division
import numpy as np
import pandas as pd
# from scipy import integrate


def _calculate_sigma(x, stability_class):
    """
    Calculates sigmay and sigmaz as a function of grid points in the 
    direction of travel (x) for stability class A through F.

    Parameters
    ---------------
    x: numpy array
        Grid points in the direction of travel (m)
    stability_class : string
        Stability class, A through F
        
    Returns
    ---------
    sigmay: numpy array
        Standard deviation of the Gaussian distribution in the horizontal
        (crosswind) direction (m)
    sigmaz: numpy array
        Standard deviation of the Gaussian distribution in the vertical
        direction (m)
    """
    if stability_class == 'A':
        k = [0.250, 927, 0.189, 0.1020, -1.918]
    elif stability_class == 'B':
        k = [0.202, 370, 0.162, 0.0962, -0.101]
    elif stability_class == 'C':
        k = [0.134, 283, 0.134, 0.0722, 0.102]
    elif stability_class == 'D':
        k = [0.0787, 707, 0.135, 0.0475, 0.465]
    elif stability_class == 'E':
        k = [0.0566, 1070, 0.137, 0.0335, 0.624]
    elif stability_class == 'F':
        k = [0.0370, 1170, 0.134, 0.0220, 0.700]
    else:
        return

    sigmay = k[0] * x / (1 + x / k[1]) ** k[2]
    sigmaz = k[3] * x / (1 + x / k[1]) ** k[4]

    return sigmay, sigmaz


def _modify_grid(model, wind_direction, wind_speed):
    """
    Rotates grid to account for wind direction.
    Translates grid to account for source location.
    
    Parameters
    ---------------
    model: chama GaussianPlume
        GaussianPlume object
    wind_direction: float
        Wind direction (degrees)
    wind_speed: float
        Wind speed (m/s)

    Returns
    ---------
    gridx: numpy array
        x values in the grid (m)
    gridy: numpy array
        y values in the grid (m)
    gridz: numpy array
        z values in the grid (m)
    """

    angle_rad = wind_direction / 180.0 * np.pi
    gridx = (model.grid.x - model.source.x) * np.cos(angle_rad) \
            + (model.grid.y - model.source.y) * np.sin(angle_rad)
    gridy = - (model.grid.x - model.source.x) * np.sin(angle_rad) \
            + (model.grid.y - model.source.y) * np.cos(angle_rad)

    gridx[gridx < 0] = 0

    gridz = _calculate_z_with_buoyancy(model, gridx, wind_speed)
        
    return gridx, gridy, gridz


def _calculate_z_with_buoyancy(model, x, wind_speed):
    """
    Adjusts grid in z direction to account for buoyancy.
    
    Parameters
    ---------------
    model: chama GaussianPlume or GaussianPuff 
        GaussianPlume or GaussianPuff object
    x: numpy array
        Distance in the downwind direction from the source (m)
    wind_speed: float
        Wind speed (m/s)

    Returns
    -----------
    z: numpy array
        z values in the grid (m)
    """
    # buoyancy_parameter units: [m^4/s^3]
    buoyancy_parameter = (model.gravity * model.source.rate / np.pi) \
                         * (1.0 / model.density_eff - 1.0 / model.density_air)

    # z units: [m]
    z = model.source.z \
        + (1.6 * (buoyancy_parameter ** (1.0 / 3)) * x ** (2.0 / 3)) \
          / wind_speed

    return z


class Grid(object):

    def __init__(self, x, y, z):
        """
        Defines the receptor grid.
        
        Parameters
        --------------
        x: numpy array
            x values in the grid (m)
        y: numpy array
            y values in the grid (m)
        z: numpy array
            z values in the grid (m)
        """
        self.x, self.y, self.z = np.meshgrid(x, y, z)


class Source(object):

    def __init__(self, x, y, z, rate):
        """
        Defines the source location and leak rate.
        
        Parameters
        -------------
        x: float
            x location of the source (m)
        y: float
            y location of the source (m)
        z: float
            z location of the source (m)
        rate: float
            source leak rate (kg/s)
        """
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate


class GaussianPlume:
    
    def __init__(self, grid, source, atm,
                 gravity=9.81, density_eff=0.769, density_air=1.225):
        """
        Defines the Gaussian plume model.
        
        Parameters
        ---------------
        grid: chama Grid
            Grid points at which concentrations should be calculated
        source: chama Source
            Source location and leak rate
        atm: pandas DataFrame 
            Atmospheric conditions for the simulation. Columns include 
            'Wind Direction', 'Wind Speed', and 'Stability Class' indexed by 
            the time that changes occur.
        gravity: float
            Gravity (m2/s), default = 9.81 m2/s
        density_eff: float
            Effective density of the leaked species (kg/m3),
            default = 0.769 kg/m3
        density_eff: float
            Effective density of air (kg/m3), default = 1.225 kg/m3
        """
        self.grid = grid
        self.source = source
        self.atm = atm
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air
        self.conc = pd.DataFrame()
        self.run()

    def run(self):
        """
        Computes the concentrations of a Gaussian plume.
        """

        conc = pd.DataFrame()
        for t in self.atm.index:
            
            wind_direction = self.atm.at[t, 'Wind Direction']
            wind_speed = self.atm.at[t, 'Wind Speed']
            stability_class = self.atm.at[t, 'Stability Class']

            X2, Y2, h = _modify_grid(self, wind_direction, wind_speed)
            sigmay, sigmaz = _calculate_sigma(X2, stability_class)
            
            a = np.zeros(X2.shape)
            b = np.zeros(X2.shape)
            c = np.zeros(X2.shape)

            a[X2 > 0] = self.source.rate / \
                    (2 * np.pi * wind_speed * sigmay[X2 > 0] * sigmaz[X2 > 0])
            b[X2 > 0] = np.exp(-Y2[X2 > 0] ** 2 / (2 * sigmay[X2 > 0] ** 2))
            c[X2 > 0] = np.exp(-(self.grid.z[X2 > 0] - h[X2 > 0]) ** 2 /
                               (2 * sigmaz[X2 > 0] ** 2)) \
                      + np.exp(-(self.grid.z[X2 > 0] + h[X2 > 0]) ** 2 /
                               (2 * sigmaz[X2 > 0] ** 2))
            
            conc_at_t = a * b * c
            conc_at_t[np.isnan(conc_at_t)] = 0
            conc_at_t = pd.DataFrame(data=np. transpose([self.grid.x.ravel(),
                self.grid.y.ravel(), self.grid.z.ravel(), conc_at_t.ravel()]), 
                columns=['X', 'Y', 'Z', 'S'])
            conc_at_t['T'] = t
            
            conc = pd.concat([conc, conc_at_t], ignore_index=True)

        self.conc = conc
        self.conc = self.conc[['X', 'Y', 'Z', 'T', 'S']]

class GaussianPuff:
    
    def __init__(self, grid=None, source=None, atm=None, tpuff=1, tend=None,
                 tstep=10, gravity=9.81, density_eff=0.769, density_air=1.225):
        """
        Defines the Gaussian puff model.  
        
        Parameters
        ---------------
        grid: chama Grid
            Grid points at which concentrations should be calculated
        source: chama Source
            Source location and leak rate
        atm: pandas DataFrame 
            Atmospheric conditions for the simulation. Columns include 
            'Wind Direction','Wind Speed', and 'Stability Class' indexed by 
            the time that changes occur.
        tpuff: float
            Time between puffs (s)
        tend: float
            Total time to run the simulation (s). Must be divisible by
            tpuff
        tstep: float
            Time step for reporting concentration information (s)
        gravity: float
            Gravity (m2/s), default = 9.81 m2/s
        density_eff: float
            Effective density of the leaked species (kg/m3),
            default = 0.769 kg/m3
        density_air: float
            Effective density of air (kg/m3), default = 1.225 kg/m3
        """
        # TODO Do keyword checks, must have atm! Can't vary the stability class

        self.grid = grid
        self.source = source
        self.atm = atm
        self.tpuff = tpuff
        self.tend = tend
        self.tstep = tstep
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air
        self.puff = pd.DataFrame()
        self.conc = pd.DataFrame()
        self._make_and_track_puffs()

        if self.grid is not None and self.source is not None:
                self.run(grid, tstep)

    def _make_and_track_puffs(self):
        """
        Generates puffs for the entire simulation time. For each puff and
        each time step the location of the puff center is tracked along
        with the total distance traveled from the source and the
        standard deviations in the horizontal and vertical directions
        (sigmaY and sigmaZ). All of this information is stored in a
        pandas DataFrame called puff.
        """

        if self.tend is None:
            self.tend = max(self.atm.index)
        tpuff = self.tpuff
        tend = self.tend

        temp = pd.DataFrame({'T': 0, 'X': self.source.x, 'Y': self.source.y,
                            'Z': self.source.z, 'D': 0, 'Puff_ID': 0}, 
                            index=[0])

        df_list = [temp]
        tprev = temp
        wind_direction = 0
        wind_speed = 0
        stability_class = None

        # This will only work when tend is divisible by tpuff
        timepoints = np.linspace(tpuff, tend, num=int(tend / tpuff),
                             endpoint=True)
        for t in timepoints:
            
            if t - tpuff in self.atm.index:
                wind_direction = self.atm.at[int(t - tpuff), 'Wind Direction']
                wind_speed = self.atm.at[int(t - tpuff), 'Wind Speed']
                stability_class = self.atm.at[int(t - tpuff),
                                               'Stability Class']
                
            # Update distances
            angle_rad = wind_direction / 180.0 * np.pi

            r = tpuff * wind_speed
            x = r * np.cos(angle_rad)
            y = r * np.sin(angle_rad)

            if abs(x) < 1E-5:
                x = 0.0
            if abs(y) < 1E-5:
                y = 0.0

            temp = tprev.copy()
            temp['X'] += x
            temp['Y'] += y
            temp['D'] += r
            temp['T'] = t
            temp['Z'] = _calculate_z_with_buoyancy(self, temp['D'].values,
                                                   wind_speed)
            # New Puff
            new_puff = pd.DataFrame({'T': t, 'X': self.source.x,
                                'Y': self.source.y, 'Z': self.source.z,
                                'D': 0, 'Puff_ID': t}, index=[0])
            temp = pd.concat([temp, new_puff], ignore_index=True)
            
            df_list.append(temp)
            tprev = temp

        puff = pd.concat(df_list, ignore_index=True)
        sigmay, sigmaz = _calculate_sigma(puff['D'], stability_class)
        puff['sigmaY'] = sigmay
        puff['sigmaZ'] = sigmaz

        self.Q = self.source.rate * tpuff
        self.puff = puff

    def run(self, grid, tstep):
        """
        Computes the concentrations of a Gaussian puff model.

        Parameters
        -----------------
        grid: chama Grid
            Grid points at which concentrations should be calculated
        tstep: float
            Time step for reporting concentration information (s)
        """
        
        self.grid = grid
        self.tstep = tstep

        times = [i * tstep for i in range(int(self.tend / tstep) + 1)]

        conc_list = []

        for t in times:
            
            # Extract the puff data at time t
            mask = (self.puff['T'] >= t - 0.1 * self.tpuff) \
                   & (self.puff['T'] <= t + 0.1 * self.tpuff)
            temp = self.puff.loc[mask].copy()  
            temp = temp.reset_index()
            conc_at_t = np.zeros(grid.x.shape)

            for i in temp.index:
                xk = temp.iloc[i].X
                yk = temp.iloc[i].Y
                zk = temp.iloc[i].Z
                sigmay = temp.iloc[i].sigmaY
                sigmaz = temp.iloc[i].sigmaZ
                if sigmay == 0 or sigmaz == 0:
                    continue
                
                if sigmay >= 8 or sigmaz >= 8:
                    continue
                
                x_part = np.exp(-((xk - grid.x)**2) / (2 * sigmay**2))
                y_part = np.exp(-((yk - grid.y)**2) / (2 * sigmay**2))
                z_part = np.exp(-((zk - grid.z)**2) / (2 * sigmaz**2))
                z_reflection = np.exp(-((zk + grid.z)**2) / (2 * sigmaz**2))
                conc_at_t += 1 / (sigmay**2 * sigmaz) * x_part * y_part * \
                             z_part

                # Add reflection part
                conc_at_t += 1 / (sigmay**2 * sigmaz) * x_part * y_part * \
                             z_reflection
            
            conc_at_t = conc_at_t * self.Q / ((2 * np.pi)**1.5)
            conc_at_t = pd.DataFrame(data=np.transpose([self.grid.x.ravel(),
                                                        self.grid.y.ravel(),
                                                        self.grid.z.ravel(),
                                                        conc_at_t.ravel()]),
                                     columns=['X', 'Y', 'Z', 'S'])
            conc_at_t['T'] = t
            conc_list.append(conc_at_t)

        self.conc = pd.concat(conc_list, ignore_index=True)
        self.conc = self.conc[['X', 'Y', 'Z', 'T', 'S']]
