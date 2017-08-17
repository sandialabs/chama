.. raw:: latex

    \newpage

.. _sensors:

Sensor technology
=================

Many different sensor technologies exist for a particular application. For
example, in the context of gas detection, sensors may monitor the
concentration at a fixed point or they may be based on optical gas imaging
technology and monitoring an area within the field of view of the sensor.
Sensors might be monitoring continuously or at defined sampling times.
Sensors could also be mounted to vehicles or drones and moving through a
detection area. In order to understand the tradeoffs between different
technologies and select an optimal subset of sensors, these different sensor
technologies should be considered simultaneously within an optimal sensor
placement problem.

In Chama, a Sensor object is used to represent a variety of sensor properties
including detector type, detection threshold, location, and sampling times.
In addition, every sensor object includes a function that accepts a signal
DataFrame and returns the subset of that signal that is detected by the
sensor at the defined sampling times. This information is then used by the
:ref:`impact` module to determine the impact of the sensor on a scenario.

The :mod:`chama.sensors` module can be used to represent 4 types of sensor
technologies; stationary and mobile point sensors and stationary and mobile
camera sensors. In addition, the module has been designed to be extensible
so that additional sensor technologies could easily be incorporated. A
sensor is declared by specifying a position and a detector. See the
:ref:`example` section for examples of declaring different sensor types.
More information about the Sensor, Position, and Detector classes can be
found in the :mod:`chama.sensors` documentation.

Position class types
--------------------

- **Position**: used to define a stationary sensor that is fixed at a single
  location.

- **Mobile**: defines a mobile sensor that moves according to defined waypoints
  and speed. It can also be defined to repeat its path.

Detector class types
--------------------

- **SimpleSensor**: used to define a concentration point detector. This
  detector type determines detection by comparing a signal to the detector's
  threshold.

- **Camera**: defines a camera detector. The camera model is based on those in
  [include references]. See the API documentation for details on the available
  camera parameters. This detector type determines detection by collecting
  the signal within the camera's field of view, converting that signal to
  pixels, and comparing that to the detector's threshold in terms of pixels
