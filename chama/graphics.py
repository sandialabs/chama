"""
The graphics module contains ...
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull  
import numpy as np

def signal_convexhull(signal, scenarios, threshold, timesteps=None,  
                   colormap=plt.cm.viridis, txyz_names=['T', 'X', 'Y', 'Z'], 
                   x_range=(None, None), y_range=(None, None), z_range=(None, None)):
    
    t_col = txyz_names[0]
    x_col = txyz_names[1]
    y_col = txyz_names[2]
    z_col = txyz_names[3]

    if timesteps is None:
        timesteps = sorted(set(signal.loc[:,t_col]))

    fig = plt.figure()
    plt.set_cmap(colormap)
    ax = fig.add_subplot(111, projection='3d')

    for scenario in scenarios:
        i = 0
        for timestep in timesteps:
            try:
                color = colormap(i)
                i = i + 1/float(len(timesteps))
                
                signal_t = signal[signal[t_col] == timestep]
                conc_filter = signal_t[scenario] > threshold
                
                # plot points
                #data = signal_t[[x_col,y_col,z_col,scenario]][conc_filter]
                #data = data.as_matrix()
                #ax.scatter(data[:,0], data[:,1], data[:,2], c=data[:,3],s=30)
                
                data = signal_t[[x_col,y_col,z_col]][conc_filter]
                data = data.as_matrix()
                hull=ConvexHull(data)
                ax.plot_trisurf(data[:,0], data[:,1], data[:,2], 
                                triangles=hull.simplices,
                                edgecolor='none', 
                                shade=False,
                                color=color)
            except:
                pass

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    
    ax.set_xlim3d(x_range[0],x_range[1])
    ax.set_ylim3d(y_range[0],y_range[1])
    ax.set_zlim3d(z_range[0],z_range[1])

def signal_xsection(signal, scenarios, threshold = None, timesteps=None, 
                        x_value=None, y_value=None, z_value=None, log_flag = False,
                        colormap=plt.cm.viridis, alpha = 0.7, V = 10,
                        txyz_names=['T', 'X', 'Y', 'Z'], 
                        x_range=(None, None), y_range=(None, None), z_range=(None, None)):
        
    t_col = txyz_names[0]
    x_col = txyz_names[1]
    y_col = txyz_names[2]
    z_col = txyz_names[3]

    if timesteps is None:
        timesteps = sorted(set(signal.loc[:,t_col]))
      
    fig = plt.figure(figsize=(20,5))
    plt.set_cmap(colormap)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    
    signal_t = signal[signal[t_col].isin(timesteps)]
    signal_t = signal_t.groupby([x_col, y_col, z_col]).mean()
    data = signal_t[scenarios]
    
    def contour_data(temp, threshold, log_flag):
        temp = temp.reset_index()
        temp.columns = ['X', 'Y', 'value']
        temp = temp.pivot('X', 'Y')
        X = temp.columns.levels[1].values
        Y = temp.index.values
        Z = temp.values
        Xi,Yi = np.meshgrid(X, Y)
        if threshold:
            Z[Z<=threshold] = threshold
        if log_flag:
            Z = np.log10(Z)
        return Xi, Yi, Z
    
    if z_value:
        temp = data.xs(z_value,level=2).mean(axis=1)
    else:
        temp = data.groupby(level=[0,1]).mean().sum(axis=1)
    Xi, Yi, Z = contour_data(temp, threshold, log_flag)
    ax1.contourf(Yi, Xi, Z, V, alpha=alpha, cmap=colormap)
    ax1.set_xlim(x_range[0],x_range[1])
    ax1.set_ylim(y_range[0],y_range[1])
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    
    if y_value:
        temp = data.xs(y_value,level=1).mean(axis=1)
    else:
        temp = data.groupby(level=[0,2]).mean().sum(axis=1)
    Xi, Yi, Z = contour_data(temp, threshold, log_flag)
    ax2.contourf(Yi, Xi, Z, V, alpha=alpha, cmap=colormap)
    ax2.set_xlim(x_range[0],x_range[1])
    ax2.set_ylim(z_range[0],z_range[1])
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(z_col)
    
    if x_value:
        temp = data.xs(x_value,level=0).mean(axis=1)
    else:
        temp = data.groupby(level=[1,2]).mean().sum(axis=1)
    Xi, Yi, Z = contour_data(temp, threshold, log_flag)
    ax3.contourf(Yi, Xi, Z, V, alpha=alpha, cmap=colormap)
    ax3.set_xlim(y_range[0],y_range[1])
    ax3.set_ylim(z_range[0],z_range[1])
    ax3.set_xlabel(y_col)
    ax3.set_ylabel(z_col)
