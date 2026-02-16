

import warnings
warnings.filterwarnings("ignore")
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cfgrib

folder_data = "/Odyssey/private/t22picar/data/mld/"
#file_glorys = "era5_2019-2021_hourly.grib" # Actually not glorys 
file_obs = "mld_2009-12-30-2022-12-28.nc"
maps = xr.open_dataset(folder_data+file_obs)
lat_obs = maps.latitude.values
lon_obs = maps.longitude.values
maps = maps.rename({"latitude": "lat"})
maps = maps.rename({"longitude": "lon"})

folder_data_4th = "/Odyssey/private/t22picar/data/ssh_L4/"
file_glorys_4th = "SSH_L4_CMEMS_2010-01-01-2024-01-01_4th.nc"
maps_4th = xr.open_dataset(folder_data_4th+file_glorys_4th).sel(time=slice("2010-01-01","2022-12-28"))

# Daily interpolation
maps_inter = maps.interp({"time": maps_4th.time}, method="linear")

# grid interpolation 
maps_inter = maps_inter.interp({"lat":maps_4th.lat, "lon":maps_4th.lon}, method="linear")

# save data 
save_file=file_obs[:-3]+"_daily_4th"+".nc"
# Sauvegarder le DataArray en fichier NetCDF
maps_inter.to_netcdf(folder_data+save_file)
