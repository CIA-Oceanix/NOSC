import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from glob import glob

# Import data to interp
# TO EDIT
#file_map = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2019.nc"

size_grid="8th"
xp_name="unet_uv_drifters_aoml_15m_10y_11d_bathy_no_sst_mae"

list_of_maps = sorted(glob(f'/Odyssey/private/t22picar/multivar_drifter/rec/{xp_name}/daily/*.nc'))
maps = xr.open_mfdataset(list_of_maps, combine='nested', concat_dim='time')

start_date = datetime(2019, 1, 1)
end_date = datetime(2020, 1, 1)

start_time= start_date.strftime("%Y-%m-%d")
end_time= end_date.strftime("%Y-%m-%d")

                                
file_out = "/Odyssey/private/t22picar/data/uv/unet/"
str_save_file = f"{xp_name}_{size_grid}.nc"

# Import reference grid 
map_4th = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01.nc"
map_4th = xr.open_dataset(map_4th).sel(time="2010-01-01")
lat_ref = map_4th.latitude.values
lon_ref = map_4th.longitude.values

map = maps.sel(time=slice(start_date,start_date))

ds = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
current_date = start_date + timedelta(days=1)

time_index=1
while current_date < end_date:
    map = maps.sel(time=current_date)
    map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

    # Concaténation
    ds = xr.concat([ds, map], dim="time")
    current_date += timedelta(days=1)
# save data 

save_file=file_out+str_save_file
# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)