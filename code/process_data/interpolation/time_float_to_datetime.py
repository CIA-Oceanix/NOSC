import xarray as xr
import numpy as np


result_filepath = "/Odyssey/private/t22picar/data/era5/era5_neu/era5_neu_2010-01-01_2022-12-31_4th.nc"

from datetime import datetime, timedelta
file_map = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01.nc"
start_date = datetime(2010, 1, 1)
end_date = datetime(2021, 12, 31)
map_4th = xr.open_dataset(file_map).sel(time=slice(start_date,end_date))
map_4th

res_data = xr.open_dataset(result_filepath)#.isel(time=slice(0,-1))
#res_data = res_data.assign_coords(time=res_data.time.values.astype("datetime64[s]").astype("datetime64[ns]"))
res_data_corr = res_data.assign_coords(time=map_4th.time)

# Sauvegarde dans un nouveau fichier
res_data_corr.to_netcdf("/Odyssey/private/t22picar/data/era5/era5_neu/era5_neu_2010-01-01_2021-12-31_4th_time.nc")