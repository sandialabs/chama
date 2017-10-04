.. raw:: latex

    \newpage

.. _sensors:

Sensor technology
=================

Many different sensor technologies exist. For example, in the context of gas
detection, sensors can monitor the concentration at a fixed point or they
can be based on optical gas imaging technology and monitor an area within a
field of view. Sensors can monitor continuously or at defined sampling
times. Sensors can also be mounted on vehicles or drones and move through
a specified region. Furthermore, sensors can have different operating
conditions which can change detectability.  In order to understand the
tradeoffs between different sensor technologies and operating conditions
and to select an optimal subset of sensors, these different options should
be considered simultaneously within an optimal sensor placement problem.

The :mod:`chama.sensors` module can be used to define sensor technologies in
Chama. The module is used to represent a variety of sensor properties
including detector type, detection threshold, location, and sampling times.
Additionally, every sensor object includes a function that accepts a `signal`, 
described in the :ref:`transport` section, and returns the subset of that
signal that is detected by a set of sensors. This information is then used
to extract the `impact` of each sensor on each scenario, as described in the
:ref:`impact` section. The sensor placement optimization uses this measure of 
'impact' to select sensors.

Each sensor is declared by specifying a **position** and a **detector**.
The following options are available in Chama (additional sensor 
technologies could easily be incorporated).

Position options
----------------

- **Stationary**: A stationary sensor that is fixed at a single location.

- **Mobile**: A mobile sensor that moves according to defined waypoints
  and speed. It can also be defined to repeat its path or start moving at a
  particular time. A mobile sensor is assumed to be at its first waypoint
  for all times before its starting time and is assumed to be at its final
  waypoint if it has completed its path and the repeat path option was not set.

Detector options
----------------

- **Point**: A point detector. This type of
  detector determines detection by comparing a signal to the detector's
  threshold.

- **Camera**: A camera detector using the camera model from [RaWB16]_. 
  This type of detector determines detection by collecting
  the signal within the camera's field of view, converting that signal to
  pixels, and comparing that to the detector's threshold in terms of pixels.
  
*When using signal data in XYZ format*, Chama can interpolate sensor measurements that are not represented 
in the signal data.  However, the sample time of a Camera detectors must be represented 
in the signal data (i.e. only X, Y, and Z can be interpolated).

For example, a **stationary point sensor**, can be defined as follows:

.. doctest::
    :hide:

    >>> import chama
	
.. doctest::

    >>> pos1 = chama.sensors.Stationary(location=(1,2,3))
    >>> det1 = chama.sensors.Point(threshold=0.001, sample_times=[0,2,4,6,8,10])
    >>> stationary_pt_sensor = chama.sensors.Sensor(position=pos1, detector=det1)

A **mobile point sensor**, can be defined as follows:

.. doctest::

    >>> pos2 = chama.sensors.Mobile(locations=[(0,0,0),(1,0,0),(1,3,0),(1,2,1)],speed=1.2)
    >>> det2 = chama.sensors.Point(threshold=0.001, sample_times=[0,1,2,3,4,5,6,7,8,9,10])
    >>> mobile_pt_sensor = chama.sensors.Sensor(position=pos2, detector=det2)

A **stationary camera sensor**, can be defined as follows:

.. doctest::

    >>> pos3 = chama.sensors.Stationary(location=(2,2,1))
    >>> det3 = chama.sensors.Camera(threshold=400, sample_times=[0,5,10], direction=(1,1,1))
    >>> stationary_camera_sensor = chama.sensors.Sensor(position=pos3, detector=det3)

A **mobile camera sensor**, can be defined as follows:

.. doctest::

    >>> pos4 = chama.sensors.Mobile(locations=[(0,1,1),(0.1,1.2,1),(1,3,0),(1,2,1)],speed=0.5)
    >>> det4 = chama.sensors.Camera(threshold=100, sample_times=[0,3,6,9], direction=(1,1,1))
    >>> mobile_camera_sensor = chama.sensors.Sensor(position=pos4, detector=det4)

*When using signal data in J format*, Chama does not interpolate sensor measurements 
that are not represented in the signal data and only stationary point sensor can be used.
When using J format, a **stationary point sensor**, can be defined as follows:

.. doctest::

    >>> pos1 = chama.sensors.Stationary(location='Node1')
    >>> det1 = chama.sensors.Point(threshold=0.001, sample_times=[0,2,4,6,8,10])
    >>> stationary_pt_sensor = chama.sensors.Sensor(position=pos1, detector=det1)

Note that the units for time, location, speed, and threshold need to match
the units from the transport simulation.