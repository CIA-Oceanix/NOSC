print("Make daily map aoml")

from glob import glob
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../../tools")
from plot_tools import plot_uv_map,plot_map_zoom
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import gzip
import numpy
import netCDF4
import datetime
import os
from typing import Optional, Tuple
from scipy import stats

from datetime import datetime, timedelta



lon_bins = np.linspace(-180.125, 179.916672 + 0.125, 1441)
lat_bins = np.linspace(-80.125, 90.125, 681)

print("Import reference grid" )
# Import reference grid 
map_4th = "/Odyssey/private/t22picar/data/glorys_15m/glorys_multivar_15m_2010-2018.nc"
map_4th = xr.open_dataset(map_4th).sel(time="2010-01-01")
lat = map_4th.lat.values
lon = map_4th.lon.values
#lon = (lon_bins[1:] + lon_bins[:-1]) / 2
#lat = (lat_bins[1:] + lat_bins[:-1]) / 2
print("Reference grid imported" )


start_date = datetime(2010, 1, 1)
end_date = datetime(2023, 1, 1)

current_date = start_date
time_index = 0
year = 0

u_drifter_list =[]
v_drifter_list =[]

print(current_date)

while current_date < end_date:
        
        check_year = current_date.year
        if check_year != year:
                print(check_year)
                year = check_year
                filenames_drifters = sorted(glob(f'/Odyssey/public/drifters/cmems_00m/drifter_*_{year}.nc'))
                ds_drifter = xr.open_mfdataset(filenames_drifters, combine='nested', concat_dim='date')
                ds_drifter = ds_drifter.sel(date=ds_drifter["date.year"] == year)
                ds_drifter = ds_drifter.compute()
                ds_drifter = ds_drifter.where(ds_drifter['ums'] < 1000, drop=True)
                ds_drifter = ds_drifter.where(ds_drifter['vms'] < 1000, drop=True) 

        # Make daily bin map with u, v mean
        day_ds_drifter = ds_drifter.sel(date = current_date.strftime("%Y-%m-%d"))
        if day_ds_drifter.ums.values.shape[0]==0:
                u_drifter = np.empty((lon.shape[0],lat.shape[0]))
                u_drifter[:] = np.nan
                v_drifter = np.empty((lon.shape[0],lat.shape[0]))
                v_drifter[:] = np.nan
        else :
            u_drifter,x_edge,y_edge,binnumber = stats.binned_statistic_2d(day_ds_drifter.lon.values, day_ds_drifter.lat.values, day_ds_drifter.ums.values, 'mean', bins=[lon_bins, lat_bins])
            v_drifter,x_edge,y_edge,binnumber = stats.binned_statistic_2d(day_ds_drifter.lon.values, day_ds_drifter.lat.values, day_ds_drifter.vms.values, 'mean', bins=[lon_bins, lat_bins])

        # Créer un DataArray pour "u"
        u_drifter_xr = xr.DataArray(
        u_drifter.T,
        dims=("lat", "lon"),
        coords={

                "lat": lat,
                "lon": lon,
        },
        name="u_drifter"
        ).expand_dims(time=[current_date])

        # Créer un DataArray pour "u"
        v_drifter_xr = xr.DataArray(
        v_drifter.T,
        dims=("lat", "lon"),
        coords={

                "lat": lat,
                "lon": lon,
        },
        name="v_drifter"
        ).expand_dims(time=[current_date])

        u_drifter_list.append(u_drifter_xr)
        v_drifter_list.append(v_drifter_xr)

        current_date += timedelta(days=1)  # Ajoute 1 an (approximation)
        time_index += 1


# Concaténer tous les DataArrays le long de la dimension "time"
u_drifter = xr.concat(u_drifter_list, dim="time")
v_drifter = xr.concat(v_drifter_list, dim="time")

# Créer le Dataset final
ds = xr.Dataset({"u_drifter": u_drifter, "v_drifter": v_drifter})

# Get the list of variable names
variable_names = list(ds.variables.keys())
variable_names.remove("time")

for var in variable_names:
    ds[var] = ds[var].astype(np.float32)

print("Saving...")
save_file=f"/Odyssey/private/t22picar/data/drifters/daily_uv/drifters_uv_00m_cmems_4th.nc"

# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)