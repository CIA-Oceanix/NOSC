
import warnings
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

size_grid="8th"
depth_str="15m"

folder_data = f"/Odyssey/private/t22picar/data/glorys_{depth_str}/"
file_glorys = f"glorys_multivar_{depth_str}_2010.nc"
maps_glorys = xr.open_dataset(folder_data+file_glorys)
maps_glorys = maps_glorys.sel(depth=maps_glorys.depth[0])

folder_grid_ref = "/Odyssey/private/t22picar/data/ssh_L4/"
file_grid_ref = "SSH_L4_CMEMS_2019.nc"
grid_ref = xr.open_dataset(folder_grid_ref+file_grid_ref).sel(time=slice("2019-01-01","2019-12-31"))# 
grid_ref = grid_ref.rename({"latitude": "lat"})
grid_ref = grid_ref.rename({"longitude": "lon"})

maps_glorys = maps_glorys.rename({"latitude": "lat"})
maps_glorys = maps_glorys.rename({"longitude": "lon"})


def regrid_da(da: xr.DataArray):
        new_da = da.interp({"lat":grid_ref.lat, "lon":grid_ref.lon}, method="linear")
        return new_da

for year in range(2011,2019):
    print(year)
    file_glorys = f"glorys_multivar_{depth_str}_{year}.nc"
    maps_glorys_i = xr.open_dataset(folder_data+file_glorys)
    maps_glorys_i = maps_glorys_i.sel(depth=maps_glorys_i.depth[0])

    print("Interpolation ... ")
    maps_glorys_i = maps_glorys_i.rename({"latitude": "lat"})
    maps_glorys_i = maps_glorys_i.rename({"longitude": "lon"})  
    maps_glorys_i = regrid_da(maps_glorys_i)
    print("Interpolation done ")
    print("Concatenation ... ")
    maps_glorys = xr.concat([maps_glorys, maps_glorys_i], dim='time')
    print("Concatenation done ")

#maps_glorys = maps_glorys.sel(depth=maps_glorys.depth[0])

# save data 
print("Saving...")
save_file=f"glorys_multivar_{depth_str}_2010-2018_{size_grid}.nc"

# Sauvegarder le DataArray en fichier NetCDF
maps_glorys.to_netcdf(folder_data+save_file)
print("Saving done")