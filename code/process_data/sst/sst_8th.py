# Add cloud to sst glorys
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

folder_data = "/Odyssey/private/t22picar/data/sst_L4/"
file_obs = "SST_L4_OSTIA_2019.nc"

maps = xr.open_dataset(folder_data+file_obs).sel(time=slice("2019-01-01","2019-12-31"))
maps = maps.rename({"latitude": "lat"})
maps = maps.rename({"longitude": "lon"})
maps = maps.rename({"analysed_sst": "thetao"})

folder_grid_ref = "/Odyssey/private/t22picar/data/ssh_L4/"
file_grid_ref = "SSH_L4_CMEMS_2019.nc"
grid_ref = xr.open_dataset(folder_grid_ref+file_grid_ref).sel(time=slice("2019-01-01","2019-12-31"))

def regrid_da(da: xr.DataArray):
        new_da = da.interp({"lat":grid_ref.latitude, "lon":grid_ref.longitude}, method="linear")
        return new_da

# Interpolation new grid
maps = regrid_da(maps)
maps.thetao.values = maps.thetao.values - 273.15

# save data 
save_file=file_obs[:-3]+f"_{size_grid}"+".nc"
# Sauvegarder le DataArray en fichier NetCDF
maps.to_netcdf(folder_data+save_file)
