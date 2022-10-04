"""
The sensor module contains classes to define point or camera sensors
that can either be stationary and mobile.

.. rubric:: Contents

.. autosummary::

    Sensor
    Position
    Stationary
    Mobile
    Detector
    Point
    Camera
"""
from __future__ import print_function, division
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from scipy import integrate
from scipy import ndimage as sn


class Sensor(object):

    def __init__(self, position=None, detector=None):
        """
        Defines a sensor object and methods to calculate detection.
    
        Parameters
        ----------
        position : chama Position
            Sensor position 
        detector : chama Detector
            Sensor detector, determines the method used to calculate detection
        """
        
        self.name = None

        if position is None or not isinstance(position, Position):
            raise ValueError('Must specify a Position object to create a '
                             'Sensor')
        self.position = position

        if detector is None or not isinstance(detector, Detector):
            raise ValueError('Must specify a Detector object to create a '
                             'Sensor')
        self.detector = detector

    def get_detected_signal(self, signal, interp_method=None, 
                            min_distance=10.0):
        """
        Returns the detected signal.
        """
        return self.detector.get_detected_signal(signal, self.position,
                                                 interp_method, min_distance)


class Position(object):
    
    def __init__(self, location=None):
        """
        Defines a sensor's position.
    
        Parameters
        ----------
        location : (x,y,z) tuple or index
            The location of the Position object defined as an (x,y,z) tuple 
            or location index, which can be a string or integer
        """
        self.location = location

    def __call__(self, time):
        """
        Returns the location at the specified time

        Parameters
        ----------
        time : int or float

        Returns
        -------
        Location tuple (x,y,z) or (j)

        """
        if isinstance(self.location, tuple):
            return tuple(self.location)
        else:
            return tuple([self.location])


class Stationary(Position):
    """
    Defines a stationary sensor's position.
    """
    pass


class Mobile(Position):
    """
    Defines a mobile sensor's position.
    A mobile position moves according to defined waypoints and speed. The
    mobile position is assumed to move in a straight line between waypoints
    and will repeat its path if needed.

    Parameters
    ----------
    locations : list of (x,y,z) tuples
        List of (x,y,z) tuples defining the waypoints of the mobile
        sensor's path.
    speed : int or float
        The speed of the mobile sensor in units consistent
        with the waypoints and sensor sample_times
    repeat : bool
        Boolean indicating if the path should repeat
    """
    def __init__(self, locations=None, speed=1, start_time=0, repeat=False):
        super(Mobile, self).__init__(locations)
        self.speed = speed
        self.start_time = start_time
        self.repeat = repeat
        self._d_btwn_locs = None
    
    def __call__(self, time):
        """
        Returns the position (x,y,z) at the specified time.

        Parameters
        ----------
        time : int or float

        Returns
        -------
        tuple
            The (x,y,z) location

        """
        # Calculate distance traveled at specified time
        delta_time = time - self.start_time
        if delta_time < 0:
            delta_time = 0
        distance = self.speed * delta_time

        temp_locs = [np.array(i) for i in self.location]
        if self.repeat:  # if path repeats
            temp_locs.append(temp_locs[0])  

        if self._d_btwn_locs is None:
            # Distances between consecutive points
            self._d_btwn_locs = \
                [np.linalg.norm(temp_locs[i] - temp_locs[i + 1])
                 for i in range(len(temp_locs) - 1)]

        if self.repeat:  # if path repeats
            while distance > sum(self._d_btwn_locs):
                distance -= sum(self._d_btwn_locs)
        else:
            if distance > sum(self._d_btwn_locs):
                distance = sum(self._d_btwn_locs)
        
        i = 0
        # Figure out which line segment
        for i, _ in enumerate(self._d_btwn_locs):
            if sum(self._d_btwn_locs[:i + 1]) >= distance:
                distance -= sum(self._d_btwn_locs[:i])
                break

        # The two waypoints defining the line segment
        loc1 = temp_locs[i]
        loc2 = temp_locs[i + 1]

        location = loc1 + (loc2 - loc1) * (distance / self._d_btwn_locs[i])

        return tuple(location)


class Detector(object):
    """
    Defines a sensor's detector.

    Parameters
    ----------
    threshold : int
        The minimum signal that can be detected by the sensor
    sample_times : list of ints or floats
        List of the sensor's sample/measurement times
    """

    def __init__(self, threshold=None, sample_times=None):
        self.threshold = threshold
        self.sample_times = sample_times
        self.sample_points = None

    def get_sample_points(self, position):
        """
        Returns the sensor sample points in the form (t,x,y,z) or (t,j)

        Parameters
        ----------
        position : chama Position
            The position of the sensor
            
        Returns
        -------
        A list of sample points in the form (t,x,y,z) or (t,j)
        """

        if self.sample_points is None:
            self.sample_points = [(t,) + position(t) for t in
                                  self.sample_times]
        return self.sample_points

    def get_detected_signal(self, signal, position, interp_method,
                            min_distance):
        """
        Returns the signal detected by the sensor.

        Parameters
        ----------
        signal : pandas DataFrame
            DataFrame with the multi-index (T, X, Y, Z) or (T, Node) and columns
            containing the concentrations for different scenarios
        position : chama Position
            The position of the sensor
        interp_method : 'linear', 'nearest', or None
            Method used to interpolate the signal if needed.  
            A value of 'linear' will use griddata to interpolate missing
            sample points. A value of 'nearest' will set the sample point to
            the nearest signal point within a minimum distance of min_distance.
            If there are no signal points within this distance then the
            signal will be set to zero at the sample point.
        min_distance : float
            The minimum distance when using the 'nearest' interp_method

        Returns
        -------
        A pandas Series with multi-index (T, Scenario) and signal values above
        the sensor threshold.

        """
        pts = self.get_sample_points(position)
        if len(pts) == 0:
            return pd.Series()
        
        signal_sample = self._get_signal_at_sample_points(signal, pts,
                                                          interp_method,
                                                          min_distance)
        if len(signal_sample) == 0:
            return pd.Series()
        
        # Reset the index
        signal_sample = signal_sample.reset_index()

        # At this point we don't need the X,Y,Z or Node columns
        if set(['X', 'Y', 'Z']) < set(list(signal_sample.columns)):
            signal_sample.drop(['X', 'Y', 'Z'], inplace=True, axis=1)
        elif set(['Node']) < set(list(signal_sample.columns)):
            signal_sample.drop(['Node'], inplace=True, axis=1)
        else:
            raise ValueError('Unrecognized signal format')
            return
            
        # Set T as the index
        signal_sample = signal_sample.set_index('T')

        # Apply threshold
        signal_sample = signal_sample[signal_sample >= self.threshold]
        if len(signal_sample) == 0:
            return pd.Series()
        
        # Name the columns so that the index is labeled after stacking
        signal_sample.columns.name = 'Scenario'
        
        # Drop Nan and stack by index
        return signal_sample.stack()

    def _get_signal_at_sample_points(self, signal, sample_points,
                                     interp_method, min_distance):
        raise NotImplementedError()


class Point(Detector):
    """
    Defines a point sensor.
    """

    def _get_signal_at_sample_points(self, signal, sample_points,
                                     interp_method, min_distance):
        """
        Returns the signal at the sensor sample points. If a sample point
        does not exist in the signal DataFrame then interpolate the signal.

        Parameters
        -----------
        signal : pandas DataFrame

        sample_points : list of tuples

        interp_method : 'linear', 'nearest', or None

        min_distance : float

        Returns
        ---------
        pandas DataFrame has a multi-index containing all of the
        sample_points and columns for each scenario with the
        concentration at each sample point

        """

        # Get subset of signal. If a sample point is not in signal then NaN
        # is inserted
        try:
            signal_subset = signal.loc[sample_points, :]
        except: # Some sample points are not in the signal
            if ['T', 'X', 'Y', 'Z'] == signal.index.names:
                signal_subset = pd.DataFrame(data=np.nan, columns=signal.columns, 
                                         index=pd.MultiIndex.from_tuples(sample_points, names=['T', 'X', 'Y', 'Z']))
            elif ['T', 'Node'] == signal.index.names:
                signal_subset = pd.DataFrame(data=np.nan, columns=signal.columns, 
                                         index=pd.MultiIndex.from_tuples(sample_points, names=['T', 'Node']))
            else:
                raise ValueError('Unrecognized signal format')
            
            sample_points_in_signal = list(set(sample_points).intersection(set(signal.index)))
            signal_subset.loc[sample_points_in_signal, :] = signal.loc[sample_points_in_signal, :]
        
        if interp_method is None:
            return signal_subset
        
        # Get the sample_points that need to be interpolated
        temp = signal_subset.isnull().any(axis=1)  # Get rows containing NaN
        interp_points = list(signal_subset[temp].index)  # Get their index

        if len(interp_points) == 0:
            return signal_subset
        
        # TODO: Revisit the distance calculation.
        # Scaling issue by including both time and xyz location in distance
        # calculation. Manually select the signal times bordering
        # interp_point times BEFORE calculating the distance? Or include a
        # time scaling parameter as an additional input?

        # get the distance between the signal points and the interp_points
        signal_points = list(signal.index)
        distdata = cdist(signal_points, interp_points)

        # Might not want to build this data frame when signal is very large
        dist = pd.DataFrame(data=distdata, index=signal.index)

        if interp_method == 'linear':

            # Loop over interp_points
            for i in range(len(dist.columns)):
                temp = dist.iloc[:, i]

                # Get the rows within dist_factor of the minimum distance
                dist_factor = 2
                temp2 = temp[temp < temp.min() * dist_factor]
                # Ensures that we get enough points to do the interpolation
                while len(temp2) < 100:
                    dist_factor += 1
                    temp2 = temp[temp < temp.min() * dist_factor]
                temp_signal = signal.loc[temp2.index, :]

                # Loop over scenarios
                for j in signal.columns:

                    interp_signal = griddata(list(temp_signal.index),
                                             list(temp_signal.loc[:, j]),
                                             interp_points[i],
                                             method=interp_method,
                                             rescale=True)
                    if np.isnan(interp_signal):
                        raise ValueError('Trying to interpolate a sample '
                                         'point outside of the signal grid. '
                                         'Make sure that all sensor '
                                         'locations are contained in the '
                                         'area spanned by the signal data.')
                    signal_subset.at[interp_points[i], j] = interp_signal

        elif interp_method == 'nearest':

            # Loop over interp_points
            for i in range(len(dist.columns)):
                temp = dist.iloc[:, i]

                if temp.min() > min_distance:
                    # Loop over scenarios
                    for j in signal.columns:
                        interp_signal = 0.0
                        signal_subset.at[interp_points[i], j] = interp_signal
                else:
                    temp2 = temp[temp < min_distance]
                    temp_signal = signal.loc[temp2.index, :]

                    # Loop over scenarios
                    for j in signal.columns:

                        interp_signal = griddata(list(temp_signal.index),
                                                 list(temp_signal.loc[:, j]),
                                                 interp_points[i],
                                                 method=interp_method,
                                                 rescale=True)

                        signal_subset.at[interp_points[i], j] = interp_signal
        else:
            raise ValueError('Unrecognized or unsupported interpolation method'
                             ' "%s" was specified. Only "linear" or "nearest" '
                             'interpolations are supported' % interp_method)

        return signal_subset


class Camera(Detector):
    """
    Defines a camera sensor.

    Parameters
    ----------
    threshold : int
        The minimum number of pixels that must detect something in order for
        the camera to detect.
    sample_times : list of ints or floats
        List of the sensor's sample/measurement times
    direction : (x, y, z) tuple
        Tuple representing the direction that the camera is pointing in (x,
        y, z) coordinates relative to the origin
    **kwds : dictionary
        Keyword arguments for setting parameter values in the camera model
    """
    
    def __init__(self, threshold=None, sample_times=None,
                 direction=(1, 1, 1), **kwds):

        super(Camera, self).__init__(threshold, sample_times)

        # Direction of the camera relative to the origin
        self.direction = direction

        # Set default camera properties

        # Transmission coefficient of air
        self.tau_air = kwds.pop('tau_air', 1)

        # Maximum distance that the camera can detect in (m)
        self.dist = kwds.pop('dist', 500.0)

        # TODO: Get descriptions of these from Arvind
        self.netd = kwds.pop('netd', 0.015)
        self.f_number = kwds.pop('f_number', 1.5)
        self.e_a = kwds.pop('e_a', 0.1)
        self.e_g = kwds.pop('e_g', 0.5)
        self.T_g = kwds.pop('T_g', 300)
        self.T_plume = kwds.pop('T_plume', 300)
        self.lambda1 = kwds.pop('lambda1', 3.2E-6)
        self.lambda2 = kwds.pop('lambda2', 3.4E-6)
        self.fov1 = kwds.pop('fov1', 24 * np.pi / 180)
        self.fov2 = kwds.pop('fov2', 18 * np.pi / 180)
        self.a_d = kwds.pop('a_d', 9.0E-10)
        self.Kav = kwds.pop('Kav', 2.191e-20)

        # Constants used in the camera model
        self.NA = 6.02E23  # Avogadro's number
        self.h = 6.626e-34  # Planck's constant [J-s]
        self.SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/m^2-K^4]
        self.c = 3.0e8  # Speed of light [m/s]
        self.k = 1.38e-23  # Boltzmann's constant [J/K]
    
    def _get_signal_at_sample_points(self, signal, sample_points,
                                     interp_method, min_distance):
        """
        Defines detection as seen by a camera sensor. Not just
        selecting/interpolating a subset of the signal DataFrame. We are using
        the CONCENTRATION signal DataFrame to calculate the PIXEL signal at the
        sample points.

        Parameters
        -----------
        signal : pandas DataFrame
            DataFrame has a multi-index with (T, X, Y, Z) points
            and each column in the frame contains concentration
            values at those points for one scenario

        sample_points : list of tuples (t,x,y,z)

        Returns
        ---------
        pandas DataFrame has a multi-index with the sensor's sample_points
        (T,X,Y,Z) and each column contains the number of pixels that
        detected something for one scenario
        """

        # TODO: Add option to specify a different camera direction at each
        # sample point
        CamDir = self.direction
        # Reset the index and set it to T
        allConc = signal.reset_index().set_index('T')

        # Create DataFrame to be returned
        newidx = pd.MultiIndex.from_tuples(sample_points,
                                           names=('T', 'X', 'Y', 'Z'))
        detected_pixels = pd.DataFrame(None, index=newidx,
                                       columns=signal.columns)

        # Check if all the sample times are time points included in the
        # signal dataframe
        signal_t = set(allConc.index)
        sample_t = set([p[0] for p in sample_points])
        if not sample_t.issubset(signal_t):
            raise ValueError('All sampling times for a camera sensor must be '
                             'contained in the signal data')

        # print('    Calculating camera signal detection')
        # TODO: Add interpolation for non-gridded or sparse signal data
        for point in sample_points:
            time = point[0]
            # print('        Time: ', time)
            CamLoc = point[1:]

            # We assume that every sample time is in the signal DataFrame
            # Extract the rows at the sample time
            Conc = allConc.loc[time, :]

            # Might want to move the below calculations to a new function to
            # avoid deeply nested for-loops. Any way to vectorize??

            # No longer need T column
            Conc = Conc.reset_index(drop=True)

            # Set and sort the index so that we can guarantee the order of
            # the rows and use numpy reshape to do the conversion to a 3D array
            Conc = Conc.set_index(['X', 'Y', 'Z'])
            Conc = Conc.sort_index()

            # Get all the unique X, Y, and Z grid points
            gridpoints = list(Conc.index)
            groupedpoints = list(zip(*gridpoints))
            X = np.unique(groupedpoints[0])
            Y = np.unique(groupedpoints[1])
            Z = np.unique(groupedpoints[2])

            # Check if signal is on a regular grid by looking at the number
            # of rows
            nx = len(X)
            ny = len(Y)
            nz = len(Z)

            if nx * ny * nz != Conc.shape[0]:
                raise ValueError('The camera sensor requires signal data '
                                 'to be on a regular grid')

            # Check to make sure X,Y,Z points are equally spaced
            xdiff = np.unique(X[1:] - X[:-1])
            ydiff = np.unique(Y[1:] - Y[:-1])
            zdiff = np.unique(Z[1:] - Z[:-1])
            if len(xdiff) > 1 or len(ydiff) > 1 or len(zdiff) > 1:
                raise ValueError('The camera sensor requires signal data to '
                                 'be equally spaced over a particular '
                                 'spatial axis (i.e. X, Y, and Z)')

            # Calculate angles (horizontal and vertical) associated with the
            # camera orientation. The vertical angle is complemented due to
            # spherical coordinate convention.
            dir1 = np.array(CamDir)
            dir2 = dir1 / (np.sqrt(dir1[0] ** 2 + dir1[1] ** 2 + dir1[2] ** 2))
            horiz = np.arccos(dir2[0])
            vert = np.arccos(dir2[2])

            # The camera has 320 X 240 pixels. To speed up computation, this
            # has been reduced proportionally to 80 X 60. The horizontal (vert)
            # field of view is divided equally among the 80 (60) horizontal
            # (vert) pixels
            # TODO: convert horiz/vert field of view degrees to parameters

            theta_h = np.linspace(horiz - np.pi / 15, horiz + np.pi / 15, 80)
            theta_v = np.linspace(vert - np.pi / 20, vert + np.pi / 20, 60)

            # factor_x, factor_y, factor_z are used later for
            # concentration-pathlength (CPL) calculations. Extrapolation to
            # calculate CPL happens in pixel-coordinates rather than real-life
            # coordinates. 'dist' is the maximum distance that the IR
            # camera can see
            Xstep, Ystep, Zstep = X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]
            factor_x = int(self.dist / Xstep)
            factor_y = int(self.dist / Ystep)
            factor_z = int((self.dist / 5) / Zstep)

            p, q = len(theta_h), len(theta_v)
            x_end = np.zeros((p, q))
            y_end = np.zeros((p, q))
            z_end = np.zeros((p, q))

            # Calculate the real-life coordinate of a point 'dist' m away for
            # each pixel orientation. This is used to calculate CPL. If 'dist'
            # goes outside the grid boundary the concentration is set to 0.
            for i in range(0, p):
                for j in range(0, q):
                    x_end[i, j] = factor_x * np.cos(theta_h[i]) * \
                                  np.sin(theta_v[j])
                    y_end[i, j] = factor_y * np.sin(theta_h[i]) * \
                                  np.sin(theta_v[j])
                    z_end[i, j] = factor_z * np.cos(theta_v[j])

            # Because calculations happen in pixel coordinates, the
            # location of the camera (start of calculation) and the
            # location of far-away point (end of calculation) is converted
            # to pixel coordinates.
            x_start = (CamLoc[0] - np.min(X)) / Xstep
            y_start = (CamLoc[1] - np.min(Y)) / Ystep
            z_start = (CamLoc[2] - np.min(Z)) / Zstep

            x_end += x_start
            y_end += y_start
            z_end += z_start

            # Calculate camera properties
            nep, tec = self._pixelprop()

            for scen in Conc.columns:
                # Extract the concentration values as a numpy array
                ppm = Conc.loc[:, scen].values
                # Reshape the concentration column as a 3D array
                ppm = ppm.reshape(nx, ny, nz)

                IntConc = np.zeros((p, q))
                CPL = np.zeros((p, q))

                # TODO: Convert this to vector operation to remove for-loops??
                for i in range(0, len(theta_h)):
                    for j in range(0, len(theta_v)):
                        # Calculate concentration pathlength (CPL)
                        IntConc[i, j] = self._pathlength(x_start, y_start,
                                                         z_start,
                                                         x_end[i, j],
                                                         y_end[i, j],
                                                         z_end[i, j], ppm)

                        CPL[i, j] = IntConc[i, j] * self.dist

                # Convert CPL to image contrast and compare it to nep
                # 1e-4 is conversion factor
                attn = CPL * self.Kav * self.NA * 1e-4
                temp = 1 - 10 ** (-attn)
                contrast = temp * np.abs(tec) * self.tau_air

                # Count the number of pixels with a contrast greater than nep
                pixels = sum(sum(contrast >= nep))

                # Camera pixels were truncated to 80 x 60 px, convert pixel
                # count back to the original scale
                pixel_final = 16 * pixels

                detected_pixels.at[point, scen] = pixel_final

        # print(detected_pixels)
        return detected_pixels

    def _pathlength(self, x0, y0, z0, x1, y1, z1, data):
        num = 201  # number of points in extrapolation
        x = np.linspace(x0, x1, num)
        y = np.linspace(y0, y1, num)
        z = np.linspace(z0, z1, num)
        concs = sn.map_coordinates(data, np.vstack((x, y, z)), order=1)
        # CPL as a fraction of total number of points in extrapolation
        avgconc = sum(concs) / num
        return avgconc

    def _pixelprop(self):
        """
        Calculates camera properties

        Returns
        -------
        nep : float
            Noise-equivalent power

        tec : float
            Temperature-emissivity contrast
        """

        T_a = self.T_g - 20

        w1g = self.h * self.c / (self.lambda2 * self.k * self.T_g)
        w2g = self.h * self.c / (self.lambda1 * self.k * self.T_g)
        n1 = 2 * np.pi * self.k ** 4 * self.T_g ** 3 / \
             (self.h ** 3 * self.c ** 2)
        temp_y1 = -np.exp(-w1g) * (720 + 720 * w1g + 360 * w1g ** 2 +
                                   120 * w1g ** 3 + 30 * w1g ** 4 +
                                   6 * w1g ** 5 + w1g ** 6)
        temp_y2 = -np.exp(-w2g) * (720 + 720 * w2g + 360 * w2g ** 2 +
                                   120 * w2g ** 3 + 30 * w2g ** 4 +
                                   6 * w2g ** 5 + w2g ** 6)
        y1 = temp_y2 - temp_y1
        y = y1 * n1
        nep = y * self.netd * self.a_d / (4 * self.f_number ** 2)

        ppixelg = self._pixel_power(self.T_g)
        ppixelp = self._pixel_power(self.T_plume)
        ppixela = self._pixel_power(T_a)

        tec = ppixelp - self.e_g * ppixelg \
              - self.e_a * (1 - self.e_g) * ppixela

        return nep, tec

    def _pixel_power(self, temp):
        """
        Calculates the the power incident on a pixel from an infinite blackbody
        emitter at a given temperature.

        Parameters
        -----------
        temp : float
            Temperature of the emitter (K)

        Returns
        ---------
        Power incident on the pixel (W)
        """

        # Calculate the nondimensional frequency limits of the sensor
        w1 = self.h * self.c / (self.lambda2 * self.k * temp)
        w2 = self.h * self.c / (self.lambda1 * self.k * temp)

        # Integrate the blackbody radiation over the frequency range
        temp_int = integrate.quad(lambda x: x ** 3 / (np.exp(x) - 1), w1, w2)

        # calculate the power incident on one camera pixel
        frac = temp_int[0] / (np.pi ** 4 / 15)
        sblaw = self.SIGMA * temp ** 4 * self.a_d
        power = (4 / np.pi) * sblaw * np.tan(self.fov1 / 2) * \
                np.tan(self.fov2 / 2)
        pixel_power = power * frac
        return pixel_power
