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
            source leak rate (kg/s)
        """
        self.x = x
        self.y = y
        self.z = z
        self.rate = rate

class _GaussianBase(object):
    
    def __init__(self, grid, source,
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        
        self.grid = grid
        self.source = source
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air
        
    def _calculate_sigma(self, X, stability_class):
        """
        Calculate sigmay and sigmaz as a function of grid points in the 
        direction of travel (X) for stability class A through F.
        
        Returns
        ---------
        sigmay: numpy array
        
        sigmaz: numpy array
        
        """
        if stability_class == 'A':
            k = [0.250, 927, 0.189, 0.1020, -1.918]
        elif stability_class == 'B':
            k = [0.202, 370, 0.162, 0.0962, -0.101]
        elif stability_class == 'C':
            k = [0.134, 283, 0.134, 0.0722, 0.102]
        elif stability_class == 'D':
            k = [0.0787, 707, 0.135, 0.0475, 0.465]
        elif stability_class == 'E':
            k = [0.0566, 1070, 0.137, 0.0335, 0.624]
        elif stability_class == 'F':
            k = [0.0370, 1170, 0.134, 0.0220, 0.700]
        else:
            return

        sigmay = k[0]*X/(1+X/k[1])**k[2]
        sigmaz = k[3]*X/(1+X/k[1])**k[4]

        return sigmay, sigmaz
    
    def _modify_grid(self, wind_direction, wind_speed):
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
        angle_rad = wind_direction/180.0 *np.pi   
        gridx = (self.grid.x - self.source.x)*np.sin(angle_rad) + (self.grid.y - self.source.y)*np.cos(angle_rad)
        gridy = (self.grid.x - self.source.x)*np.cos(angle_rad) - (self.grid.y - self.source.y)*np.sin(angle_rad) 

        buoyancy_parameter = (self.gravity*self.source.rate/np.pi)*(1.0/self.density_eff-1.0/self.density_air) #[m^4/s^3]
        gridz = self.source.z + (1.6*(buoyancy_parameter**(1./3))*(gridx)**(2./3))/wind_speed #[m]  
        
        return gridx, gridy, gridz

class GaussianPlume(_GaussianBase):
    
    def __init__(self, grid, source,
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        """
        Guassian plume model.  
        
        Parameters
        ---------------
        grid: chama.transport.Grid object
        
        source: chama.transport.Source object

        gravity: float
            Gravity (m2/s), default = 9.81 m2/s
        
        density_eff: float
            Effective denisty of the leaked species (kg/m3), default = 0.769 kg/m3 
        
        density_eff: float
            Effective denisty of air (kg/m3), default = 1.225 kg/m3
        """
        super(GaussianPlume, self).__init__(grid, source, gravity, density_eff, density_air)
        self.grid = grid
        self.source = source
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air

    def run(self, atm):
        """
        Computes the concentrations of a gaussian plume.
        """
        conc = pd.DataFrame()
        for t in atm.index:
            
            wind_direction = atm.loc[t,'Wind Direction']
            wind_speed = atm.loc[t,'Wind Speed']
            stability_class = atm.loc[t, 'Stability Class']

            X2, Y2, h = self._modify_grid(wind_direction, wind_speed)
            sigmayX2, sigmazX2 = self._calculate_sigma(X2, stability_class)
            
            a = self.source.rate / (2 * np.pi * wind_speed * sigmayX2 * sigmazX2) 
            b = np.exp(-Y2**2/(2*sigmayX2**2))
            c = np.exp(-(self.grid.z-h)**2/(2*sigmazX2**2)) + np.exp(-(self.grid.z+h)**2/(2*sigmazX2**2))
            
            conc_at_t = a*b*c
            conc_at_t[np.isnan(conc_at_t)] = 0
            conc_at_t = pd.DataFrame(data=np.transpose([self.grid.x.ravel(),
                self.grid.y.ravel(), self.grid.z.ravel(), conc_at_t.ravel()]), 
                columns=['X', 'Y', 'Z', 'C'])
            conc_at_t['T'] = t
            
            conc = conc.append(conc_at_t, ignore_index=True)
            
        return conc
        
class GaussianPuff(_GaussianBase):
    
    def __init__(self, grid, source, 
                 gravity = 9.81, density_eff=0.769, density_air=1.225):
        """
        Guassian puff model.  
        
        Parameters
        ---------------
        grid: chama.transport.Grid object
        
        source: chama.transport.Source object

        gravity: float
            Gravity (m2/s), default = 9.81 m2/s
        
        density_eff: float
            Effective denisty of the leaked species (kg/m3), default = 0.769 kg/m3 
        
        density_eff: float
            Effective denisty of air (kg/m3), default = 1.225 kg/m3
        """
        super(GaussianPuff, self).__init__(grid, source, gravity, density_eff, density_air)
        self.grid = grid
        self.source = source
        self.gravity = gravity
        self.density_eff = density_eff
        self.density_air = density_air

    def run(self, atm):
        """
        Computes the concentrations of a gaussian puff model.
        """
        conc = pd.DataFrame()
        previous_sources = pd.DataFrame()
        for t in atm.index:
            conc_at_t = np.zeros_like(self.grid.x,dtype=float)
            
            wind_direction = atm.loc[t,'Wind Direction']
            wind_speed = atm.loc[t,'Wind Speed']
            stability_class = atm.loc[t, 'Stability Class']

            X2, Y2, h = self._modify_grid(wind_direction, wind_speed)
            sigmayX2, sigmazX2 = self._calculate_sigma(X2, stability_class)
        
            # Transport previous puffs
            for source in previous_sources.index:
                pass

                # Rotate/Translate those sources
                # Something like this? but with each previous source
                #angle_rad = wind_direction/180.0 *np.pi   
                #gridx = (self.grid.x - self.source.x)*np.sin(angle_rad) + (self.grid.y - self.source.y)*np.cos(angle_rad)
                #gridy = (self.grid.x - self.source.x)*np.cos(angle_rad) - (self.grid.y - self.source.y)*np.sin(angle_rad) 
        
                # Run Gaussian dispersion model for previous sources, rate = 0?
                # Something like this?
                #a = self.source.rate / (2 * np.pi * wind_speed * sigmayX2 * sigmazX2) 
                #b = np.exp(-Y2**2/(2*sigmayX2**2))
                #c = np.exp(-(self.grid.z-h)**2/(2*sigmazX2**2)) + np.exp(-(self.grid.z+h)**2/(2*sigmazX2**2))
                #conc_at_t += a*b*c
                
                # This loop should be vectorized
            
            # Transport current puff
            a = self.source.rate / (2 * np.pi * wind_speed * sigmayX2 * sigmazX2) 
            b = np.exp(-Y2**2/(2*sigmayX2**2))
            c = np.exp(-(self.grid.z-h)**2/(2*sigmazX2**2)) + np.exp(-(self.grid.z+h)**2/(2*sigmazX2**2))  
            conc_at_t += a*b*c
            
            conc_at_t[np.isnan(conc_at_t)] = 0
            conc_at_t = pd.DataFrame(data=np.transpose([self.grid.x.ravel(),
                self.grid.y.ravel(), self.grid.z.ravel(), conc_at_t.ravel()]), 
                columns=['X', 'Y', 'Z', 'C'])
            conc_at_t['T'] = t
            
            previous_sources = conc_at_t[conc_at_t['C'] > 0]
    
            conc = conc.append(conc_at_t, ignore_index=True)

        return conc
    