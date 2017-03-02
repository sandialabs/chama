"""
The transport module contains Gaussian air dispersion models.
"""
import numpy as np
import pandas as pd
from scipy import integrate

class Grid(object):

    def __init__(self,x,y,z):
        """
        Define the receptor grid
        
        Parameters
        --------------
        x: numpy array
            x values in the grid (m)
        y: numpy array
            y values in the grid (m)
        z: numpy array
            z values in the grid (m)
        """
        self.x,self.y,self.z = np.meshgrid(x,y,z)

class Source(object):

    def __init__(self, x,y,z,rate):
        """
        Defines the source location and leak rate
        
        Parameters
        -------------
        x: float
            x location of the source (m)
        y: float
            y location of the source (m)
        z: float
            z location of the source (m)
        rate: float
            source leak rate (g/s)
        """
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate

class _GaussianBase(object):
    
    def __init__(self, grid, source, stability_class, wind_direction,
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        
        self.grid = grid
        self.source = source
        self.stability_class = stability_class
        self.wind_direction = wind_direction
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air
        
    def _calculate_sigma(self, X):
        """
        Calculate sigmay and sigmaz as a function of grid points in the 
        direction of travel (X) for stability class A through F.
        
        Returns
        ---------
        sigmay: numpy array
        
        sigmaz: numpy array
        
        """
        if self.stability_class == 'A':
            k = [0.250, 927, 0.189, 0.1020, -1.918]
        elif self.stability_class == 'B':
            k = [0.202, 370, 0.162, 0.0962, -0.101]
        elif self.stability_class == 'C':
            k = [0.134, 283, 0.134, 0.0722, 0.102]
        elif self.stability_class == 'D':
            k = [0.0787, 707, 0.135, 0.0475, 0.465]
        elif self.stability_class == 'E':
            k = [0.0566, 1070, 0.137, 0.0335, 0.624]
        elif self.stability_class == 'F':
            k = [0.0370, 1170, 0.134, 0.0220, 0.700]
        else:
            return

        sigmay = k[0]*X/(1+X/k[1])**k[2]
        sigmaz = k[3]*X/(1+X/k[1])**k[4]

        return sigmay, sigmaz
    
    def _modify_grid(self):
        """
        Rotate grid to account for wind direction.
        Translate grid to account for source location.
        Adjust grid to account for buoyancy.
        
        Returns
        ---------
        gridx: numpy array
        
        gridy: numpy array
        
        gridz: numpy array
        """
        angle_rad = self.wind_direction/180.0 *np.pi   
        gridx = (self.grid.x - self.source.x)*np.sin(angle_rad) + (self.grid.y - self.source.y)*np.cos(angle_rad)
        gridy = (self.grid.x - self.source.x)*np.cos(angle_rad) - (self.grid.y - self.source.y)*np.sin(angle_rad) 

        buoyancy_parameter = (self.gravity*self.source.rate/np.pi)*(1.0/self.density_eff-1.0/self.density_air) #[m^4/s^3]
        gridz = self.source.z + (1.6*(buoyancy_parameter**(1./3))*(gridx)**(2./3))/self.wind_speed #[m]  
        
        return gridx, gridy, gridz

class GaussianPlume(_GaussianBase):
    
    def __init__(self, grid, source, stability_class, wind_speed, wind_direction,
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        """
        Guassian plume model.  
        
        Parameters
        ---------------
        grid: chama.transport.Grid object
        
        source: chama.transport.Source object
        
        stability_class: string
            Wind stability class, letter between A and F
        
        wind_speed: float
            Wind speed (m/s)
        
        wind_direction: float
            Wind direction (degrees)
        
        gravity: float
            Gravity (m2/s), default = 9.81
        
        density_eff: float
            Effective denisty of the leaked species (kg/m3), default = 0.769 kg/m3 
        
        density_eff: float
            Effective denisty of air (kg/m3), default = 1.225 kg/m3
        """
        super(GaussianPlume, self).__init__(grid, source, stability_class, wind_direction, gravity, density_eff, density_air)
        self.grid = grid
        self.source = source
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air

    def run(self, current_time=None, pandas_output_format=True):
        """
        Computes the concentrations of a gaussian plume.
        """
        X2, Y2, Z2 = self._modify_grid()
        sigmayX2, sigmazX2 = self._calculate_sigma(X2)
        
        a = self.source.rate / (2 * np.pi * self.wind_speed * sigmayX2 * sigmazX2) 
        b =  np.exp(-Y2**2/(2*sigmayX2**2))
        c = np.exp(-(self.grid.z-Z2)**2/(2*sigmazX2**2)) + np.exp(-(self.grid.z+Z2)**2/(2*sigmazX2**2))
        conc = a*b*c

        conc[np.isnan(conc)] = 0
        
        if pandas_output_format:
            conc = pd.DataFrame(data=np.transpose([self.grid.x.ravel(),
                       self.grid.y.ravel(), self.grid.z.ravel(), conc.ravel()]), 
                       columns=['X', 'Y', 'Z', 'C'])
            if current_time:
                conc['T'] = current_time

        return conc
        
class GaussianPuff(_GaussianBase):
    
    def __init__(self, grid, source, stability_class, wind_speed, wind_direction,
                 windstep, timestep, 
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        """
        Guassian puff model.  
        
        Parameters
        ---------------
        grid: chama.transport.Grid object
        
        source: chama.transport.Source object
        
        stability_class: string
            Wind stability class, letter between A and F
        
        wind_speed: float
            Wind speed (m/s)
        
        wind_direction: float
            Wind direction (degrees)
        
        windstep: int
        
        timestep: int
        
        gravity: float
            Gravity (m2/s), default = 9.81
        
        density_eff: float
            Effective denisty of the leaked species (kg/m3), default = 0.769 kg/m3 
        
        density_eff: float
            Effective denisty of air (kg/m3), default = 1.225 kg/m3
        """
        super(GaussianPuff, self).__init__(grid, source, stability_class, wind_direction, gravity, density_eff, density_air)
        self.grid = grid
        self.source = source
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.windstep = windstep
        self.timestep = timestep
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air

    def run(self, current_time, pandas_output_format=True):
        """
        Computes the concentrations of a gaussian puff model at a given time.
        """
        X2, Y2, Z2 = self._modify_grid()
        sigmayX2, sigmazX2 = self._calculate_sigma(X2)
        
        if np.mod(current_time,self.windstep)!=0:
            times = np.mod(current_time,self.windstep)
        else:
            times = self.windstep
            
        Qt = self.source.rate*self.timestep
        a = Qt/(2*np.pi*sigmayX2*sigmazX2)**1.5
        a[a==np.inf]=0
        b = np.exp(-Y2**2/(2*sigmayX2**2))
        c = np.exp(-(self.grid.z-Z2)**2/(2*sigmazX2**2)) + np.exp(-(self.grid.z+Z2)**2/(2*sigmazX2**2))
        
        c1 = 2*sigmayX2*sigmazX2
        pp, qq, rr = X2.shape
        time_int = np.array([integrate.quad(lambda t: np.exp(-(X2[i,j,k]-self.wind_speed*t)**2/c1[i,j,k]),0,times) \
                             for i in range(0,pp) for j in range(0,qq) for k in range(0,rr)])
        conc_int = np.reshape(time_int[:,0],X2.shape)
        conc = a*b*conc_int*c
        
        conc[np.isnan(conc)] = 0
             
        if pandas_output_format:
            conc = pd.DataFrame(data=np.transpose([self.grid.x.ravel(),
                   self.grid.y.ravel(), self.grid.z.ravel(),conc.ravel()]), 
                   columns=['X', 'Y', 'Z', 'C'])
            conc['T'] = current_time

        return conc
    