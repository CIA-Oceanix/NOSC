import numpy as np 
import os
from utils_data import interpolation_glorys_raw_data,interpolation_era5_raw_data
# Define a reference grid :
import xarray as xr

size_grid = "8th"

"""
coef = 1
lon_ref = np.linspace(-180,180,1440*coef+1)[:-1]
lat_ref = np.linspace(-90,90,720*coef+1)[:-1]
"""

map_xth = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01.nc"
map_xth = xr.open_dataset(map_xth).sel(time="2010-01-01")
lat_ref = map_xth.latitude.values
lon_ref = map_xth.longitude.values

def regrid_da(da: xr.DataArray):
        new_da = da.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
        return new_da

# Define depth
depth=15

# Define a period of training : 
year_start = 2010
year_end = 2019

# Create a directory

folder_data = "/Odyssey/private/t22picar/data/era5/"

"""
folder_data = f"/Odyssey/private/t22picar/data/train_glorys_{depth}_{size_grid}/"
if not os.path.exists(folder_data):
    os.makedirs(folder_data)
    print(f"Le dossier '{folder_data}' a été créé.")
"""

# Interpolation of era5 data for train and eval period

interpolation_era5_raw_data(year_start,year_end,folder_data,lon_ref,lat_ref,size_grid)