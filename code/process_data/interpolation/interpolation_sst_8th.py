import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

size_grid="8th"
# Import reference grid 
map_xth = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01.nc"
map_xth = xr.open_dataset(map_xth).sel(time="2010-01-01")
lat_ref = map_xth.latitude.values
lon_ref = map_xth.longitude.values

# Import data to interp
# TO EDIT
file_map = "/Odyssey/private/t22picar/data/sst_L4/SST_L4_OSTIA_2010-01-01-2022-01-01.nc"
start_date = datetime(2010, 1, 1)
end_date = datetime(2022, 1, 1)
str_save_file = f"_{size_grid}.nc"

map = xr.open_dataset(file_map).sel(time=slice(start_date,start_date))
map = map.rename({"longitude" : "lon"})
map = map.rename({"latitude" : "lat"})
map = map.rename({"analysed_sst" : "thetao"})
#map

# Get the list of variable names
variable_names = list(map.variables.keys())
variable_names.remove("time")

for var in variable_names:
    map[var] = map[var].astype(np.float32)

ds = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

#start_date = datetime(2010, 1, 2)
#end_date = datetime(2019, 1, 1)
current_date = start_date + timedelta(days=1)
time_index=1
# Boucle temporelle pour ajouter des données
while current_date < end_date:

    map = xr.open_dataset(file_map).sel(time=current_date)
    map = map.rename({"longitude" : "lon"})
    map = map.rename({"latitude" : "lat"})
    map = map.rename({"analysed_sst" : "thetao"})
    map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

    # Get the list of variable names
    variable_names = list(map.variables.keys())
    variable_names.remove("time")

    for var in variable_names:
        map[var] = map[var].astype(np.float32)

    # Concaténation
    ds = xr.concat([ds, map], dim="time")
    current_date += timedelta(days=1)

#map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

# save data 
save_file=file_map[:-3]+str_save_file
# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)