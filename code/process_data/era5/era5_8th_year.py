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

size_grid="8th"
year = sys.argv[1]

print(f"year = {year}")

folder_data = "/Odyssey/private/t22picar/data/era5/"
file_glorys = f"era5_{year}.grib" # Actually not glorys 

maps_glorys = xr.open_dataset(folder_data+file_glorys, engine="cfgrib")
maps_glorys['longitude'] = xr.where(maps_glorys['longitude'] > 180, maps_glorys['longitude'] - 360, maps_glorys['longitude'])
lat_simu = maps_glorys.latitude.values
lon_simu = maps_glorys.longitude.values
maps_glorys = maps_glorys.rename({"latitude": "lat"})
maps_glorys = maps_glorys.rename({"longitude": "lon"})

folder_grid_ref = "/Odyssey/private/t22picar/data/ssh_L4/"
file_grid_ref = "SSH_L4_CMEMS_2019.nc"
grid_ref = xr.open_dataset(folder_grid_ref+file_grid_ref).sel(time=slice("2019-01-01","2019-12-31"))
grid_ref = grid_ref.rename({"latitude": "lat"})
grid_ref = grid_ref.rename({"longitude": "lon"})

def regrid_da(da: xr.DataArray):
        new_da = da.interp({"lat":grid_ref.lat, "lon":grid_ref.lon}, method="linear")
        return new_da

# Daily mean wind
print("computing daily mean ...")
maps_glorys = maps_glorys.resample(valid_time='1D').mean()
print("computation daily mean done")
# Interpolation new grid
print("interpolating ...")
maps_glorys = regrid_da(maps_glorys)
print("interpolation done")

#Split into two files
maps_glorys = maps_glorys.drop("number").drop("step").drop("surface").rename({"valid_time": "time"})

# save data 
print("Saving file ...")
save_file=f"era5_{year}_dailymean_{size_grid}.nc"

# Sauvegarder le DataArray en fichier NetCDF
maps_glorys.to_netcdf(folder_data+save_file)
print("Saving done")