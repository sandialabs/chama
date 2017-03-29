.. raw:: latex

    \newpage

Sensor technology
==========================

The :mod:`chama.sensors` module has more information on setting up stationary, mobile, and camera sensors.

Stationary
------------
A stationary sensor is fixed at a single location. 
Sensor measurements are captured at set sampling times.
The sensor returns measurements at a single point or includes a 
camera and returns measurements which include a field of view. 

Mobile 
-----------
A mobile sensor moves according to defined waypoints and speed.  
Sensor measurements are captured at set sampling times.  
The mobile sensor can be defined to repeat the path.
Based on the sensor location at a given time, the sensor returns measurements 
at a single point or includes a camera and returns measurements which include 
a field of view.  

Camera
--------
