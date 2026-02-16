import numpy as np 
import os
from utils_data import interpolation_ssh_raw_data,interpolation_sst_raw_data,interpolation_era5_raw_data_test
import xarray as xr
# Define a reference grid :

size_grid = "4th"
coef = 1
lon_ref = np.linspace(-180,180,1440*coef+1)[:-1]
lat_ref = np.linspace(-90,90,720*coef+1)[:-1]

def regrid_da(da: xr.DataArray):
        new_da = da.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
        return new_da

# Define depth
depth=15

# Define a period of test : 
year = 2019

# Create a directory

folder_data = f"/Odyssey/private/t22picar/data/train_glorys_{depth}_{size_grid}/"
if not os.path.exists(folder_data):
    os.makedirs(folder_data)
    print(f"Le dossier '{folder_data}' a été créé.")
# 

# SSH 
interpolation_ssh_raw_data(year,depth,folder_data,lon_ref,lat_ref)

# SST 

interpolation_sst_raw_data(year,depth,folder_data,lon_ref,lat_ref)

#ERA? 
interpolation_era5_raw_data_test(folder_data,lon_ref,lat_ref)