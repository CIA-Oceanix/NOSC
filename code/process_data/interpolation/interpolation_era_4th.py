import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from netCDF4 import Dataset
import pandas as pd

size_grid="4th"
# Import reference grid 
map_xth = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01_4th.nc"
start_date = "2010-01-01"
end_date = "2022-12-31"
map_xth = xr.open_dataset(map_xth).sel(time=slice(start_date,end_date))
nt, ny, nx = map_xth.dims["time"], map_xth.dims["lat"], map_xth.dims["lon"]

lat_ref = map_xth.lat.values
lon_ref = map_xth.lon.values

# Import data to interp
# TO EDIT
file_map = "/Odyssey/private/t22picar/data/era5/era5_neu/"
str_save_file = f"era5_neu_{start_date}_{end_date}_{size_grid}.nc"

# save data 
output_file=file_map+str_save_file

with Dataset(output_file, "w", format="NETCDF4") as nc:
    # Créer les dimensions
    nc.createDimension("time", nt)
    nc.createDimension("lat", ny)
    nc.createDimension("lon", nx)

    # Créer les variables
    times = nc.createVariable("time", "f8", ("time",))
    lat = nc.createVariable("lat", "f4", ("lat",))
    lon = nc.createVariable("lon", "f4", ("lon",))
    u10 = nc.createVariable("u10", "f4", ("time", "lat", "lon"))
    v10 = nc.createVariable("v10", "f4", ("time", "lat", "lon"))

    # Écrire les coordonnées fixes
    lat[:] = map_xth["lat"].values
    lon[:] = map_xth["lon"].values
    times[:] = map_xth["time"].values

    print(output_file)
    year_0=0
    # Boucle d’écriture incrémentale
    for i, t in enumerate(map_xth.time):
        year= pd.to_datetime(t.values).year
        if year!=year_0:
            print(year)
            year_0=year
        file_era = f"era5_neu_{year}.grib" # Actually not glorys 
        #print("Opening era file...")
        subset = xr.open_dataset(file_map+file_era, engine="cfgrib").sel(time=pd.to_datetime(t.values).strftime('%Y-%m-%d'))
        subset['longitude'] = xr.where(subset['longitude'] > 180, subset['longitude'] - 360, subset['longitude'])
        subset = subset.resample(valid_time='1D').mean()
        subset = subset.interp({"latitude":lat_ref, "longitude":lon_ref}, method="linear")

        if "u10" in subset:
            print("u10")
            u10[i, :, :] = subset["u10"].values
            v10[i, :, :] = subset["v10"].values

        if "u10n" in subset:
            print("u10n")
            u10[i, :, :] = subset["u10n"].values
            v10[i, :, :] = subset["v10n"].values

print("change time value")

res_data_corr = xr.open_dataset(output_file)
res_data_corr = res_data_corr.assign_coords(time=res_data_corr.time.astype("datetime64[s]"))
res_data_corr = res_data_corr.assign_coords(time=map_xth.time)

# Sauvegarde dans un nouveau fichier
res_data_corr.to_netcdf(file_map+str_save_file[-3]+".nc")