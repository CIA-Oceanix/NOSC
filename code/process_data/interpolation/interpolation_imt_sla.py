import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from glob import glob
#from jaxparrow import cyclogeostrophy, geostrophy

# Import data to interp
# TO EDIT
#file_map = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2019.nc"

# Date_limit
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 1, 1)
start_time= start_date.strftime("%Y-%m-%d")
end_time= end_date.strftime("%Y-%m-%d")

size_grid="4th"
# Output name               
file_out = "/Odyssey/public/UNET_rec_sla/mapping/"
str_save_file = f"Unet_SSH_{start_time}_{end_time}_{size_grid}.nc"


#list_of_maps = sorted(glob('/Odyssey/public/UNET_rec_sla/l4_products_sla_2010_2019/4dvar-unet-to4th*.nc'))
list_of_maps = sorted(glob('/Odyssey/public/UNET_rec_sla/mapping/unet+*_TP.nc'))
maps = xr.open_mfdataset(list_of_maps, combine='nested', concat_dim='time')
maps = maps.rename({"longitude" : "lon"})
maps = maps.rename({"latitude" : "lat"})
maps.coords['lon'] = (maps.coords['lon'] + 180) % 360 - 180
maps = maps.sortby(maps.lon)
maps = maps.transpose('time','lat','lon')

lon = maps.lon.values
lat = maps.lat.values
lon2D, lat2D = np.meshgrid(lon, lat)

# Add mdt :
mdt =  "/Odyssey/public/duacs/2019/from-datachallenge-global-ose-2023/MDT_DUACS_0.25deg.nc"
mdt = xr.open_dataset(mdt)
mdt = mdt.rename({"longitude" : "lon"})
mdt = mdt.rename({"latitude" : "lat"})
mdt.coords['lon'] = (mdt.coords['lon'] + 180) % 360 - 180
mdt = mdt.sortby(mdt.lon)
mdt=mdt.interp(lat=lat,lon=lon)

# SLA --> SSH
mdt_3D = np.repeat(mdt.mdt.values[np.newaxis,:,:],maps.sla.values.shape[0],axis=0)
maps.sla.values = maps.sla.values + mdt_3D
maps = maps.rename({"sla" : "zos"})



# Import reference grid 
map_4th = "/Odyssey/private/t22picar/data/glorys_15m/glorys_multivar_15m_2010-2018.nc"
map_4th = xr.open_dataset(map_4th).sel(time="2010-01-01")
lat_ref = map_4th.lat.values
lon_ref = map_4th.lon.values

#Just first day
map = maps.sel(time=slice(start_date,start_date))
ds = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
ds["zos"] = ds["zos"].astype("float32")
current_date = start_date + timedelta(days=1)

time_index=1
while current_date < end_date:
    map = maps.sel(time=current_date)
    map = map.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
    map["zos"] = map["zos"].astype("float32")
    # Concaténation
    ds = xr.concat([ds, map], dim="time")
    current_date += timedelta(days=1)
# save data 

save_file=file_out+str_save_file
# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)