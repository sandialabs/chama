"""
Sensor properties
"""
import pandas as pd
import numpy as np

class _Base(object):
    def __init__(self):
        self.cost = 1.0
        self.failure_rate = 0.0
        self.threshold = 0
        self.sampling_times = []
        self.camera = None
        
class Point(_Base):
    """
    Point (or Stationary?) sensor class.  
    A stationary sensor returns a measurment at a single point or includes a 
    camera and returns measurements which include a field of view.  
    """
    def __init__(self, location = None):
        super(Point,self).__init__()
        self.location = location

    def get_sample_points(self):
        sample_points = []
        # Gather field of view from a camera
        if self.camera is not None:
            locations = self.camera.get_sample_points(self.location)
        else:
            locations = [self.location]
        # Append sample points 
        for t in self.sampling_times:
            for loc in locations:
                sample_points.append((t, loc[0], loc[1], loc[2]))
        return sample_points
    
    def integrate_detected_signal(self, detected):
        return _integrate_detected_signal(detected)
        
#class Line(_Base):
#
#    def __init__(self, locations=None, discretize=1):
#        super(Line,self).__init__()
#        self.locations = locations
#        self.discretize = discretize
#        
#    def get_sample_points(self):
#        sample_points = []
#        for i in range(len(self.locations)-1):
#            a = np.array(self.locations[i])
#            b = np.array(self.locations[i+1])
#            dist = np.linalg.norm(a-b)
#            inc = float(self.discretize)/dist
#            for t in self.sampling_times:
#                for i in np.arange(0,1,inc):
#                    location = a + (b-a)*i
#                    locations = [location]
#                    # Append sample points 
#                    for loc in locations:
#                        sample_points.append((t, loc[0], loc[1], loc[2]))
#        return sample_points
#    
#    def integrate_detected_signal(self, detected):
#        return _integrate_detected_signal(detected)
        
class Path(_Base):
    """
    Path (or Mobile?) sensor class.  
    A mobile sensor moves according to a defined path and speed.  
    Measurments are captured at set sampling times.  
    The mobile sensor can be defined to repeat the path.
    Based on the sensors location at a given time, the sensor returns measurments 
    at a single point or includes a camera and returns measurments which include a field of view.  
    """
    def __init__(self, locations=None, speed=1, repeat=False):
        super(Path,self).__init__()
        self.locations = locations
        self.speed = speed
        self.repeat = repeat
    
    def get_sample_points(self):
        # Sort sampling times
        self.sampling_times.sort()
        repeat = self.repeat
        sample_points = []
        current_time = 0
        
        # Add start point to the end of the list of repeat = True
        if repeat:
            self.locations.append(self.locations[0])
            
        while True:
            for i in range(len(self.locations)-1): #grab each segment, define all sample points in that interval
                if current_time >= self.sampling_times[-1]:
                    repeat=False
                    break # break out of 'for i in range(len(self.locations)-1)' loop
                a = np.array(self.locations[i])
                b = np.array(self.locations[i+1])
                dist = np.linalg.norm(a-b)
                inc = float(self.speed)/dist
                sample_times = [t-current_time for t in self.sampling_times if t > current_time]
                temp_current_time = current_time
                for t in sample_times:
                    if self.speed*t <= dist:
                        location = a + (b-a)*inc*t
                        if self.camera is not None:
                            locations = self.camera.get_sample_points(location)
                        else:
                            locations = [location]
                        for loc in locations:
                            sample_points.append((t+temp_current_time, loc[0], loc[1], loc[2]))
                        current_time = t+temp_current_time
                    else:
                        current_time = dist/self.speed+temp_current_time
                        break # break out of 'for t in sample_times' loop
            if not repeat:
                break
           
        return sample_points
    
    def integrate_detected_signal(self, detected):
        return _integrate_detected_signal(detected)
        
class Camera(object):
    def __init__(self, position=(0,0,0), direction=(0,0,-1), top=(0,1,0), hfov=5.5*np.pi/180, vhov=4.4*np.pi/180, npixels=20):
        super(Camera,self).__init__()
        self.position = position
        self.direction = direction
        self.top = top
        self.npixels = npixels
        self.hfov = hfov
        self.vfov = vhov
        
        # Normalize and compute right side orientation of the camera
        self.direction = self.direction/np.sqrt(np.dot(self.direction,self.direction))
        self.top = self.top/np.sqrt(np.dot(self.top,self.top))
        self.right = np.cross(self.direction,self.top);
    
        # Pixel maker
        x=np.linspace(-self.hfov/2,self.hfov/2,self.npixels)
        y=np.linspace(-self.vfov/2,self.vfov/2,self.npixels)
        xgrid, ygrid = np.meshgrid(x,y)
        self.theta=np.sqrt(np.power(xgrid,2)+np.power(ygrid,2))
        xgrid[xgrid < 0] = xgrid[xgrid < 0] + np.pi
        self.phi = np.arctan(ygrid/xgrid)
        
        # Image width
        self.image_width = np.tan(self.hfov/2)*self.position(2)*2;
        
    def get_sample_points(self):
        sample_points = []
        # To Do
        return sample_points
        
    def integrate_detected_signal(self, detected):
        # To Do
        return detected

def _integrate_detected_signal(detected):
    # Other integration methods are possible.  This one just uses sum
    integrated_detection = detected.groupby(level=0).sum()
    return integrated_detection
