import numpy as np 
import xarray as xr

num_diff = 2  # Usual value is 2
num_centered = num_diff // 2
# Gravitation parameter
constant_g = 9.81
# Coriolis parameter
constant_f0 = 2 * 7.2921e-5
# Convert degrees to meters
degtom = 111.11e3
#Mask around equator
lat_mask = 10

def compute_uvgeo_3D(ssh,lat2D,lon2D): # For ssh 3D

    # 
    ugeo_l = []
    vgeo_l = []
    
    for time_index in range(ssh.shape[0]):

        ssh_i = ssh[time_index,:]
        # Mask invalid SSH values
        ssh_i[abs(ssh_i) > 100] = np.nan

        # Initialize null matrices
        ugeo = np.full(np.shape(ssh_i), 0.)
        vgeo = np.full(np.shape(ssh_i), 0.)

        # Compute g/f
        goverf = constant_g / (constant_f0 * np.sin(np.deg2rad(lat2D)))

        # Compute derivatives
        dsshy = ssh_i[num_diff:, :] - ssh_i[:-num_diff, :]
        dsshx = ssh_i[:, num_diff:] - ssh_i[:, :-num_diff]
        dlonx = (
            (lon2D[:,num_diff:] - lon2D[:,:-num_diff])
            * np.cos(np.deg2rad(lat2D[:, num_centered:-num_centered]))
            * degtom
        )
        dlaty = (lat2D[num_diff:,:] - lat2D[:-num_diff,:]) * degtom

        # Compute geostrophic velocities
        ugeo[num_centered:-num_centered, :] = -goverf[num_centered:-num_centered, :] * dsshy / dlaty
        vgeo[:, num_centered:-num_centered] = goverf[:, num_centered:-num_centered] * dsshx / dlonx

        # Mask invalid values 
        #ugeo = np.ma.array(ugeo, mask=((abs(ugeo)>10)))
        #vgeo = np.ma.array(vgeo, mask=((abs(vgeo)>10)))

        #Mask close to the equator
        ugeo_l.append( ugeo )
        vgeo_l.append( vgeo )

    ugeo_l, vgeo_l = np.array(ugeo_l), np.array(vgeo_l)

    return(ugeo_l,vgeo_l)


def compute_uvgeo(ssh,lat2D,lon2D): # For ssh 2D

    # 
    #ssh = maps.zos.values[time_index,:]


    # Mask invalid SSH values
    ssh[abs(ssh) > 100] = np.nan

    # Initialize null matrices
    ugeo = np.full(np.shape(ssh), 0.)
    vgeo = np.full(np.shape(ssh), 0.)

    # Compute g/f
    goverf = constant_g / (constant_f0 * np.sin(np.deg2rad(lat2D)))

    # Compute derivatives
    dsshy = ssh[num_diff:, :] - ssh[:-num_diff, :]
    dsshx = ssh[:, num_diff:] - ssh[:, :-num_diff]
    dlonx = (
        (lon2D[:,num_diff:] - lon2D[:,:-num_diff])
        * np.cos(np.deg2rad(lat2D[:, num_centered:-num_centered]))
        * degtom
    )
    dlaty = (lat2D[num_diff:,:] - lat2D[:-num_diff,:]) * degtom

    # Compute geostrophic velocities
    ugeo[num_centered:-num_centered, :] = -goverf[num_centered:-num_centered, :] * dsshy / dlaty
    vgeo[:, num_centered:-num_centered] = goverf[:, num_centered:-num_centered] * dsshx / dlonx

    # Mask invalid values 
    #ugeo = np.ma.array(ugeo, mask=((abs(ugeo)>10)))
    #vgeo = np.ma.array(vgeo, mask=((abs(vgeo)>10)))

    #Mask close to the equator
    ugeo = np.ma.array(ugeo, mask=((abs(lat2D)<10)))
    vgeo = np.ma.array(vgeo, mask=((abs(lat2D)<10)))

    return(ugeo,vgeo)