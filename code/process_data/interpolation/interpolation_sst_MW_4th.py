import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from glob import glob

# Import data to interp
# TO EDIT
size_grid="4th"
file_map = "/Odyssey/private/t22picar/data/sst_L4/SST_L4_OSTIA_2010-01-01-2022-01-01.nc"
start_date = datetime(2010, 1, 1)
end_date = datetime(2022, 1, 1)
str_save_file = f"_{size_grid}.nc"

folder_data="/Odyssey/private/t22picar/data/sst_L4/MW_2010_2020"
list_of_maps = sorted(glob(f'{folder_data}/*.nc'))
maps = xr.open_mfdataset(list_of_maps, combine='nested', concat_dim='time')

# Import reference grid 
map_4th = "/Odyssey/private/t22picar/data/glorys_15m/glorys_multivar_15m_2010-2018.nc"
map_4th = xr.open_dataset(map_4th).sel(time="2010-01-01")
lat_ref = map_4th.lat.values
lon_ref = map_4th.lon.values

"""
map = maps.sel(time=slice(start_date,start_date))
map = map.rename({"analysed_sst" : "thetao"})
#map

ds = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

#start_date = datetime(2010, 1, 2)
#end_date = datetime(2019, 1, 1)
current_date = start_date + timedelta(days=1)
time_index=1
# Boucle temporelle pour ajouter des données
while current_date < end_date:

    map = maps.sel(time=current_date)
    map = map.rename({"analysed_sst" : "thetao"})
    map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

    # Concaténation
    ds = xr.concat([ds, map], dim="time")
    current_date += timedelta(days=1)
"""

maps = maps.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

# save data 
save_file="/Odyssey/private/t22picar/data/sst_L4/SST_MW_2010_2020_4th.nc"
# Sauvegarder le DataArray en fichier NetCDF
maps.to_netcdf(save_file)