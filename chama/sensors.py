"""
Sensor properties
"""
import pandas as pd
import numpy as np
import time

class _Base(object):
    def __init__(self):
        self.cost = 1.0
        self.failure_rate = 0.0
        self.camera = None
        self.threshold = 0
        self.sampling_times = []

class Point(_Base):
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
        
    def integrate_detected_signal2(self, detected, txyz_names=['T', 'X', 'Y', 'Z']):
        return _integrate_detected_signal2(detected, txyz_names=txyz_names)
        
class Line(_Base):
    def __init__(self, locations=None, discretize=1):
        super(Line,self).__init__()
        self.locations = locations
        self.discretize = discretize
        
    def get_sample_points(self):
        sample_points = []
        for i in range(len(self.locations)-1):
            a = np.array(self.locations[i])
            b = np.array(self.locations[i+1])
            dist = np.linalg.norm(a-b)
            inc = float(self.discretize)/dist
            for t in self.sampling_times:
                for i in np.arange(0,1,inc):
                    location = a + (b-a)*i
                    # Gather field of view from a camera
                    if self.camera is not None:
                         locations = self.camera.get_sample_points(location)
                    else:
                        locations = [location]
                    # Append sample points 
                    for loc in locations:
                        sample_points.append((t, loc[0], loc[1], loc[2]))
        return sample_points
    
    def integrate_detected_signal(self, detected):
        return _integrate_detected_signal(detected)
        
class Path(_Base):
    def __init__(self, locations=None, speed=1):
        super(Path,self).__init__()
        self.locations = locations
        self.speed = speed
    
    def get_sample_points(self):
        # This needs work
        sample_points = []
        remainder = 0
        t = 0
        for i in range(len(self.locations)-1):
            a = np.array(self.locations[i])
            b = np.array(self.locations[i+1])
            dist = np.linalg.norm(a-b)
            inc = float(self.speed)/dist
            for i in np.arange(remainder,1+inc, inc):
                location = a + (b-a)*i
                if t in self.sampling_times:
                    # Gather field of view from a camera
                    if self.camera is not None:
                         locations = self.camera.get_sample_points(location)
                    else:
                        locations = [location]
                    # Append sample points 
                    for loc in locations:
                        sample_points.append((t, loc[0], loc[1], loc[2]))
                    t = t+1
            remainder = np.linalg.norm(location-b)
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

def _integrate_detected_signal2(detected, txyz_names=['T', 'X', 'Y', 'Z']):
    # Other integration methods are possible.  This one just uses sum
    integrated_detection = detected.groupby(by=txyz_names[0]).sum()
    del integrated_detection[txyz_names[1]]
    del integrated_detection[txyz_names[2]]
    del integrated_detection[txyz_names[3]]
    return integrated_detection