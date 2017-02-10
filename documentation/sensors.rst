.. raw:: latex

    \newpage

Sensor technology
==========================

Stationary
------------
A stationary sensor is fixed at a single location. 
Sensor measurements are captured at set sampling times.
The sensor returns measurements at a single point or includes a 
camera and returns measurements which include a field of view. 

See :meth:`~chama.sensors.Stationary` for more details.

Mobile 
-----------
A mobile sensor moves according to defined waypoints and speed.  
Sensor measurements are captured at set sampling times.  
The mobile sensor can be defined to repeat the path.
Based on the sensor location at a given time, the sensor returns measurements 
at a single point or includes a camera and returns measurements which include 
a field of view.  

See :meth:`~chama.sensors.Mobile` for more details.

Camera
--------
:meth:`~chama.sensors.Camera`