.. raw:: latex

    \newpage

.. _sensors:

Sensor technology
=================

Many different sensor technologies exist. For
example, in the context of gas detection, sensors can monitor the
concentration at a fixed point or they can be based on optical gas imaging
technology and monitoring an area within the field of view of the sensor.
Sensors can monitoring continuously or at defined sampling times.
Sensors can also be mounted to vehicles or drones and move through a
specified region. In order to understand the tradeoffs between different
technologies and select an optimal subset of sensors, these different sensor
technologies should be considered simultaneously within an optimal sensor
placement problem.

The :mod:`chama.sensors` module can be used to define sensor technologies in Chama.
The module is used to represent a variety of sensor properties
including detector type, detection threshold, location, and sampling times.
Additionally, every sensor object includes a function that accepts a `signal`, 
described in the :ref:`transport` section, and returns the subset of that 
signal that is detected by a set of sensors. This information is then used to extract
the `impact` of each sensor on each scenario, as described in the :ref:`impact` section.
This information is used as input to the sensor placement optimization.

Each sensor is declared by specifying a **position** and a **detector**.
Currently, four types of sensor technologies can be defined
(additional sensor technologies could easily be incorporated):

* Stationary point sensors

* Mobile point sensors

* Stationary camera sensors

* Mobile camera sensors

Position options
--------------------

- **Stationary**: A stationary sensor that is fixed at a single location.

- **Mobile**: A mobile sensor that moves according to defined waypoints
  and speed. It can also be defined to repeat its path.

Detector options
--------------------

- **Point**: A point detector. This type of
  detector determines detection by comparing a signal to the detector's
  threshold.

- **Camera**: A camera detector using the camera model from [RaWB16]_. 
  This type of detector determines detection by collecting
  the signal within the camera's field of view, converting that signal to
  pixels, and comparing that to the detector's threshold in terms of pixels.

