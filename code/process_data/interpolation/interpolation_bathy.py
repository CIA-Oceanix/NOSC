import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np

import xarray as xr

file_map = "/Odyssey/public/glorys/bathymetry/bathymetry_4th.nc"
file_map_output = "/Odyssey/private/t22picar/data/bathy/bathymetry_4th_cmems.nc"

# Import reference grid 
map_4th = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01_4th.nc"
map_4th = xr.open_dataset(map_4th).sel(time="2010-01-01")
lat_ref = map_4th.lat.values
lon_ref = map_4th.lon.values

map = xr.open_dataset(file_map)

map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

# Sauvegarder le DataArray en fichier NetCDF
map.to_netcdf(file_map_output)