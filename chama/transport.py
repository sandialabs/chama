import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.close('all')

def gaussian_plume(leak_flux, wind_speed, wind_direction, sampling_locations, leak_points = [0.0,0.0,0.0], density_eff = 0.769, density_air = 1.225,):
    """
    Computes the concentrations of a gaussian plume given sampling locations. 
    The gaussian plume equation is taken from the Stanford FEAST documentation.
    TO-DO: Determine standard deviation equations for various stability classes
    and distances from the source.
    
    Parameters
    ----------
    leak_flux : float
        Float to quatify the leak flux in kg/s
        
    wind_speed : float
        Float to quantify the wind speed in m/s

    wind_direction : float
        Float to quantify the wind direction in degrees azimuth
    
    sampling_locations : pd.DataFrame
        Pandas DataFrame containing the x, y, and z coordinates of sampling 
        locations.
        Index = sampling location number, Rows = x, y, z (sampling coordinates)
        Sampling locations are given in meters  
        
    leak_points : array-like
        x, y, z coordinates for the leak source in meters
        default value = [0.0, 0.0, 0.0]
    
    density_eff : float
        density of the effluent (i.e. methane)
        default value = 0.769 kg/m^3
    
    density_air : float
        density of air
        default value = 1.225 kg/m^3
              
    Returns
    -------
    plume_results : pd.Dataframe
        Pandas DataFrame with the calculated concentrations organized by 
        sampling location. In same format as sampling_locations DataFrame with
        an additional column that details concentrations.
          
    
    References
    ----------
    [1] Stanford FEAST
    """
    # Constants
    GRAVITY = 9.8 #[m/s^2] 
    I_y = 0.18 #[ ]
    I_z = 0.06 #[ ]    
    
    x = sampling_locations.x
    y = sampling_locations.y
    z = sampling_locations.z    
    
    source_x = leak_points[0] #[m]    
    source_y = leak_points[1] #[m]
    source_z = leak_points[2] #[m]    
    
    # Rotate grid to 
    angle_rad = wind_direction/180.0 *np.pi        
    x_rotated = (x-source_x)*np.sin(angle_rad) + (y-source_y)*np.cos(angle_rad)
    y_rotated = (x-source_x)*np.cos(angle_rad) - (y-source_y)*np.sin(angle_rad) 
                  
    # Calculate standard deviation of plume (function of distance downwind)              
    std_plume_conc_y = I_y * x_rotated #[m]
    std_plume_conc_z = I_z * x_rotated #[m]       
    
    # Calculate gaussian plume
    buoyancy_parameter = (GRAVITY*leak_flux/np.pi)*(1.0/density_eff-1.0/density_air) #[m^4/s^3]
    
    middle_plume_height = source_z + (1.6*(buoyancy_parameter**(1./3))*(x_rotated - source_x)**(2./3))/wind_speed #[m]     
                 
    concentration = (leak_flux/(2.0*np.pi*std_plume_conc_z*std_plume_conc_y*wind_speed))\
                        *np.exp(-0.5*(y_rotated-source_y)**2.0/std_plume_conc_y**2.0)\
                        *(np.exp(-0.5*(z-middle_plume_height)**2.0/std_plume_conc_z**2.0)+np.exp(-0.5*(z+2.0*middle_plume_height)**2.0/std_plume_conc_z**4.0))                    
    
    # Replace NaN values with zeros (necessary for sampling points upwind of plume source)                    
    concentration = np.nan_to_num(concentration)
    
    # Add concentrations to sampling locations
    plume_results = sampling_locations
    plume_results['conc'] = concentration       
    
    return plume_results
    
    
def main():
    num_samples = 5000

    X_MIN = 0
    X_MAX = 10
    
    Y_MIN = -10
    Y_MAX = 10
    
    Z_MIN = 0
    Z_MAX = 30

    x = np.random.uniform(low = X_MIN, high = X_MAX, size = num_samples)
    y = np.random.uniform(low = Y_MIN, high = Y_MAX, size = num_samples)
    z = np.random.uniform(low = Z_MIN, high = Z_MAX, size = num_samples)

    sampling_locations =  np.transpose(np.vstack((x,y,z)))
    
    sampling_locations_df = pd.DataFrame(sampling_locations, columns = ['x','y','z'])

    leak_flux = 10
    wind_speed = 1 #m/s
    wind_direction = 45 #degrees

    plume_results = gaussian_plume(leak_flux, wind_speed, wind_direction, sampling_locations_df,leak_points = [2.0,3.0,5.0])

    return plume_results
    
    
plume_results = main()
plume_results['high_conc_bool'] = plume_results['conc'] > 0.001
plume_results['high_conc'] = plume_results['high_conc_bool'] * plume_results['conc']
plume_results['high_conc'].replace(to_replace=0, value=np.nan, inplace=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plume_results.x, plume_results.y, plume_results.z, c=plume_results.high_conc, marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
