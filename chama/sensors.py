"""
The sensor module contains methods to define stationary and mobile 
sensors along with camera properties.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import time


class Sensor(object):

    @classmethod
    def CameraSensor(cls, **kwds):
        return Sensor(sensor_type=Camera(), **kwds)

    @classmethod
    def MobileSensor(cls, **kwds):
        return Sensor(position=Mobile(), **kwds)

    @classmethod
    def MobileCameraSensor(cls, **kwds):
        return Sensor(position=Mobile(), sensor_type=Camera(), **kwds)

    def __init__(self, position=None, sensor_type=None, sample_times=None,
                 location=None, threshold=None):

        self.name = None

        if position:
            self.position = position
            if location:
                self.position.location = location
        else:
            self.position = Position(location=location)

        if sensor_type:
            self.sensor_type = sensor_type
            if sample_times:
                self.sensor_type.sample_times = sample_times
        else:
            self.sensor_type = SimpleSensor(sample_times=sample_times,
                                            threshold=threshold)

    def get_detected_signal(self, signal):

        return self.sensor_type.get_detected_signal(signal, self.position)


class Position(object):

    def __init__(self, location=None):

        self.location = location

    def __call__(self, time):
        """
            Return the position (x,y,z) at the specified time
        """
        return tuple(self.location)


class Mobile(Position):
    """
    Mobile position class.
    A mobile position moves according to defined waypoints and speed. The
    mobile position is assumed to repeat its path.
    """
    def __init__(self, locations=None, speed=1):
        super(Mobile, self).__init__(locations)
        self.speed = speed
        self._d_btwn_locs = None
    
    def __call__(self, time):
        """
            Return the position (x,y,z) at the specified time
        """
        # Calculate distance traveled at specified time
        distance = self.speed * time

        temp_locs = [np.array(i) for i in self.location]
        temp_locs.append(temp_locs[0])  # Assuming path repeats

        if self._d_btwn_locs is None:
            # Distances between consecutive points
            self._d_btwn_locs = \
                [np.linalg.norm(temp_locs[i] - temp_locs[i + 1])
                 for i in range(len(temp_locs) - 1)]

        while distance > sum(self._d_btwn_locs):
            distance -= sum(self._d_btwn_locs)

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


class SimpleSensor(object):

    def __init__(self, threshold=None, sample_times=None):
        self.threshold = threshold
        self.sample_times = sample_times
        self.sample_points = None

    def get_sample_points(self, position):
        """
        Generates the sensor sample points in the form (t,x,y,z)

        Parameters
        ----------
        position: position object for the sensor

        Returns
        -------
        sample_points: list of tuples

        """

        if self.sample_points is None:
            self.sample_points = [(t,) + position(t) for t in
                                  self.sample_times]
        return self.sample_points

    def get_detected_signal(self, signal, position):
        # Given a signal dataframe with index (T, X, Y, Z)
        # Return the detected scenarios at each sample time

        pts = self.get_sample_points(position)

        signal_sample = self._get_signal_at_sample_points(signal, pts)

        # Reset the index
        signal_sample = signal_sample.reset_index()

        # At this point we don't need the X,Y,X columns
        signal_sample.drop(['X', 'Y', 'Z'], inplace=True, axis=1)

        # Set T as the index
        signal_sample = signal_sample.set_index('T')

        # Apply threshold
        signal_sample = signal_sample[signal_sample > self.threshold]

        # Drop Nan and stack by index
        return signal_sample.stack()

    def _get_signal_at_sample_points(self, signal, sample_points):
        """
        Extract the signal at the sensor sample points. If a sample point
        does not exist in the signal DataFrame then interpolate the signal

        Parameters
        -----------
        signal : pd.DataFrame

        sample_points : list of tuples

        Returns
        ---------
        signal_subset : pd.DataFrame

        """

        # Get subset of signal. If a sample point is not in signal then NaN
        # is inserted
        signal_subset = signal.loc[sample_points, :]

        # Get the sample_points that need to be interpolated
        temp = signal_subset.isnull().any(axis=1)  # Get rows containing NaN
        interp_points = list(signal_subset[temp].index)  # Get their index

        if len(interp_points) == 0:
            return signal_subset

        print('Interpolation required for ', len(interp_points), ' points')
        t0 = time.time()
        # TODO: Revisit the distance calculation.
        # Scaling issue by including both time and xyz location in distance
        # calculation. Manually select the signal times bordering
        # interp_point times BEFORE calculating the distance?

        # get the distance between the signal points and the interp_points
        signal_points = list(signal.index)
        distdata = cdist(signal_points, interp_points)

        # Might not want to build this data frame when signal is very large
        dist = pd.DataFrame(data=distdata, index=signal.index)

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
            # print('   # points used in interpolation: ', len(temp_signal))

            # Loop over scenarios
            for j in signal.columns:

                interp_signal = griddata(list(temp_signal.index),
                                         temp_signal.loc[:, j],
                                         interp_points[i],
                                         method='linear')
                signal_subset.loc[interp_points[i], j] = interp_signal

        print('   Interpolation time: ', time.time()-t0, ' sec')

        return signal_subset


class Camera(SimpleSensor):
    """
    Defines a camera sensor
    """

    def __init__(self, threshold=None, sample_times=None,
                 direction=(1, 1, 1), **kwds):
        super(Camera, self).__init__(threshold, sample_times)

        # Direction of the camera represented by a point on the unit circle
        self.direction = direction

        # Set default camera properties

        # Transmission coefficient of air
        self.tau_air = kwds.pop('tau_air', 1)

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



    def get_detected_signal(self, signal, position):
        NA = 6.02E23  # Avogadro's number
        h = 6.626e-34  # Planck's constant [J-s]
        SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/m^2-K^4]
        c = 3e8  # Speed of light [m/s]
        k = 1.38e-23  # Boltzmann's constant [J/K]
        CamDir = self.direction

        for time in self.sample_times:
            CamLoc = position(time)

        # if Conc is None:
        #     Conc = np.ones((18491,
        #                     4)) * 10e-3  # Example numbers: Uniform
        #     # concentration of about 15 ppm
        #     # Conc = np.random.lognormal(mean=1.0, sigma=1.0, size=(50,50,10))
        #
        # if X is None:
        #     X = np.linspace(-200, 200, 41)  # Default X values
        #
        # if Y is None:
        #     Y = np.linspace(-200, 200, 41)  # Default Y values
        #
        # if Z is None:
        #     Z = np.linspace(0, 10, 11)  # Default Z values

        nx, ny, nz = np.size(X), np.size(Y), np.size(Z)

        ppm = np.reshape(Conc[:, 3], (nx, ny, nz),
                         order='C')  # reshaping concentration column as a
        # 3D array

        # ---------Calculating angles (horizontal and vertical) associated
        # with camera orientation. The vertical angle
        #         is complemented due to spherical coordinate convention.-------------------------------------------

        dir1 = np.array(CamDir) - np.array(CamLoc)
        dir2 = dir1 / (np.sqrt(dir1[0] ** 2 + dir1[1] ** 2 + dir1[2] ** 2))
        horiz = np.arccos(dir2[0])
        vert = np.pi / 2 - np.arccos(dir2[2])

        # --------The camera has 320 X 240 pixels. To speed up computation,
        # this has been reduced proportionally to 80 X 60.
        #        The horizontal (vert) field of view is divided equally among the 80 (60) horizontal (vert) pixels.

        theta_h = np.linspace(horiz - np.pi / 15, horiz + np.pi / 15, 80)
        theta_v = np.linspace(vert - np.pi / 20, vert + np.pi / 20, 60)

        # -------factor_x, factor_y, factor_z are used later for concentration-pathlength (CPL) calculations. This is because
        #       extrapolation to calculate CPL happens in pixel-coordinates rather than real-life coordinates. The value 500
        #       is used as a proxy for a large distance. Beyond 500 m, the IR camera doesn't see anything.

        Xstep, Ystep, Zstep = X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]
        factor_x, factor_y, factor_z = int(500 / Xstep), int(500 / Ystep), int(
            100 / Zstep)

        p, q = len(theta_h), len(theta_v)
        x_end, y_end, z_end = np.zeros((p, q)), np.zeros((p, q)), np.zeros(
            (p, q))

        # -------Here, we calculate the real-life coordinate of a far-away point (say, 500 m away) for each pixel orientation.
        #       This is used to calculate CPL. If 500 m goes outside the boundary of 3D considered, concentration is 0.

        for i in range(0, p):
            for j in range(0, q):
                x_end[i, j] = factor_x * np.cos(theta_h[i]) * np.cos(
                    theta_v[j])
                y_end[i, j] = factor_y * np.sin(theta_h[i]) * np.cos(
                    theta_v[j])
                z_end[i, j] = factor_z * np.cos(theta_h[i])

                # -------Because calculations happen in pixel coordinates,
                # the location of the camera (start of calculation) and the
                #       location of far-away point (end of calculation) is converted to pixel coordinates.

        shiftx, shifty, shiftz = (CamLoc[0] - np.min(X)) / Xstep, (
        CamLoc[1] - np.min(Y)) / Ystep, (CamLoc[2] - np.min(Z)) / Zstep

        x_start, y_start, z_start = CamLoc[0] / Xstep + shiftx, CamLoc[
            1] / Ystep + shifty, CamLoc[2] / Zstep + shiftz
        x_end, y_end, z_end = x_end + shiftx, y_end + shifty, z_end + shiftz

        # ------These are real-space coordinates of the end points used in simulation
        Rx_end, Ry_end, Rz_end = (x_end + 1 - shiftx) * Xstep, (
        y_end + 1 - shifty) * Ystep, (z_end + 1 - shiftz) * Zstep

        # ------Used to calculate camera properties including noise-equivalent power (nep), temperature-emissivity contrast
        #      (tec) and absorption coefficient (Kav). Temperature is assumed to be 300 K, with an emissivity of 0.5.
        camprop = cp.pixelprop(300, 300)
        nep, tec, Kav = camprop[0], camprop[1], camprop[2]

        IntConc, dist, CPL = np.zeros((p, q)), np.zeros((p, q)), np.zeros(
            (p, q))

        # ------This is where concentration pathlength (CPL) is calculated using properties of images.
        for i in range(0, len(theta_h)):
            for j in range(0, len(theta_v)):
                IntConc[i, j] = cp.Pathlength(x_start, y_start, z_start,
                                              x_end[i, j], y_end[i, j],
                                              z_end[i, j], ppm)
                dist[i, j] = np.sqrt((Rx_end[i, j] - CamLoc[0]) ** 2 + (
                Ry_end[i, j] - CamLoc[1]) ** 2 + (
                                     Rz_end[i, j] - CamLoc[2]) ** 2)
                CPL[i, j] = IntConc[i, j] * dist[i, j]

                # -------This section converts CPL to image contrast and
                # compares it to nep.
        attn = CPL * Kav * NA * 1e-4  # 1e-4 is conversion factor
        temp = 1 - 10 ** (-attn)
        contrast = temp * np.abs(tec) * tau_air

        pixels = 0
        for i in range(0, len(theta_h)):
            for j in range(0, len(theta_v)):
                if contrast[i, j] >= nep:
                    pixels = pixels + 1

        pixel_final = 16 * pixels  # Camera pixels were initially truncated to 80 x 60 px, which is re-converted.

        detect = 0
        if pixel_final >= 400:
            detect = 1

        return detect

    def _pathlength(x0, y0, z0, x1, y1, z1, data):
        num = 201  # number of points in extrapolation
        x, y, z = np.linspace(x0, x1, num), np.linspace(y0, y1, num), np.linspace(
            z0, z1, num)
        concs = sn.map_coordinates(data, np.vstack((x, y, z)))
        test = sum(
            concs) / num  # CPL as a fraction of total number of points in
        # extrapolation
        return test


    def _pixelprop(T_g, T_plume):
        # Camera Properties assumed as default
        netd = 0.015
        f_number = 1.5
        e_a = 0.1
        e_g = 0.5

        if T_g is None:
            T_g = 300

        if T_plume is None:
            T_plume = 300

        T_a = T_g - 20

        w1g = h * c / (lambda2 * k * T_g)
        w2g = h * c / (lambda1 * k * T_g)
        n1 = 2 * np.pi * k ** 4 * T_g ** 3 / (h ** 3 * c ** 2)
        temp_y1 = -np.exp(-w1g) * (
        720 + 720 * w1g + 360 * w1g ** 2 + 120 * w1g ** 3 + 30 * w1g ** 4 + 6 * w1g ** 5 + w1g ** 6)
        temp_y2 = -np.exp(-w2g) * (
        720 + 720 * w2g + 360 * w2g ** 2 + 120 * w2g ** 3 + 30 * w2g ** 4 + 6 * w2g ** 5 + w2g ** 6)
        y1 = temp_y2 - temp_y1
        y = y1 * n1
        nep = y * netd * a_d / (4 * f_number ** 2)

        ppixelg = pixel_power(T_g)
        ppixelp = pixel_power(T_plume)
        ppixela = pixel_power(T_a)

        tec = ppixelp - e_g * ppixelg - e_a * (1 - e_g) * ppixela

        Kav = 2.191e-20

        return nep, tec, Kav


    def _pixel_power(temp):
        """
        Calculate the the power incident on a pixel from an infinite blackbody emitter at a given temperature.
        Inputs:
            temp    Temperature of the emitter (K)
        Return:
            pixel_power   power incident on the pixel (W)
        """
        # Calculate the nondimensional frequency limits of the sensor
        w1 = h * c / (lambda2 * k * temp)
        w2 = h * c / (lambda1 * k * temp)
        # Integrate the blackobdy radiation over the frequency range
        temp_int = integrate.quad(lambda x: x ** 3 / (np.exp(x) - 1), w1, w2)
        # calculate the power incident on one camera pixel
        frac = temp_int[0] / (np.pi ** 4 / 15)
        sblaw = SIGMA * temp ** 4 * a_d
        power = (4 / np.pi) * sblaw * np.tan(FoV1 / 2) * np.tan(FoV2 / 2)
        pixel_power = power * frac
        return pixel_power

