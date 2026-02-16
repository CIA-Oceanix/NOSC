# Compute uvgeo from SSH
# Import necessary modules
import matplotlib.pyplot as plt
import numpy
import os
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import sys
sys.path.append("/Odyssey/private/t22picar/tools")
from plot_tools import plot_uv_map, plot_map_zoom_only
from datetime import datetime, timedelta
from jaxparrow import cyclogeostrophy, geostrophy
from tqdm import tqdm  # Importer tqdm
#from IPython.display import clear_output

lat_mask=5
depth = "15m"

file_data = f"/Odyssey/private/t22picar/data/glorys_{depth}/glorys_multivar_{depth}_2010-2018.nc"
maps_4th = xr.open_dataset(file_data)
#maps=maps.sel(time="2010-01-01")
lon_4th = maps_4th.lon.values
lat_4th = maps_4th.lat.values

file_data = f"/Odyssey/private/t22picar/data/glorys_{depth}/glorys_multivar_15m_2010.nc"
maps = xr.open_dataset(file_data)
#maps=maps.sel(depth=maps.depth[0])
maps=maps.sel(depth=maps.depth[0]).sel(time="2010-01-01")
maps = maps.rename({"latitude": "lat"})
maps = maps.rename({"longitude": "lon"})
maps_4th

# Get longitudes and latitudes
lon = maps.lon.values
lat = maps.lat.values
lon2D, lat2D = numpy.meshgrid(lon, lat)

maps_4th = maps_4th.drop_vars("mlotst")
maps_4th = maps_4th.rename({"zos": "uageo"})
maps_4th = maps_4th.rename({"thetao": "vageo"})
maps_4th = maps_4th.rename({"uo": "ugeo"})
maps_4th = maps_4th.rename({"vo": "vgeo"})


start_date = datetime(2010, 1, 1)
end_date = datetime(2019, 1, 1)
current_date = start_date
time_index=0
ugeo_list=[]
vgeo_list=[]
uageo_list=[]
vageo_list=[]
time_list=[]

# Initialiser la barre de progression
total_iterations = 3287
pbar = tqdm(total=total_iterations, desc="Ajout des pas de temps")

while current_date < end_date:
    print(current_date)

    year=current_date.year
    file_data = f"/Odyssey/private/t22picar/data/glorys_{depth}/glorys_multivar_15m_{year}.nc"
    maps = xr.open_dataset(file_data)
    maps=maps.sel(depth=maps.depth[0]).sel(time=current_date)

    #Compute geostrophy
    (u_geo,v_geo,lat_u, lon_u, lat_v, lon_v) = geostrophy(maps.zos.values, lat2D, lon2D)


    # Créer un DataArray pour "u"
    u_geo_xr = xr.DataArray(
        u_geo,
        dims=("lat", "lon"),
        coords={

            "lat": lat_u[:,0],
            "lon": lon_u[0,:],
        },
        name="ugeo"
    ).expand_dims(time=[current_date])

    # Créer un DataArray pour "u"
    v_geo_xr = xr.DataArray(
        v_geo,
        dims=("lat", "lon"),
        coords={

            "lat": lat_u[:,0],
            "lon": lon_u[0,:],
        },
        name="vgeo"
    ).expand_dims(time=[current_date])

    #print("Interpolation grille d'origine")
        # Interpolation grille d'origine
    u_geo_xr_int = u_geo_xr.interp({"lat":lat, "lon":lon}, method="linear")
    v_geo_xr_int = v_geo_xr.interp({"lat":lat, "lon":lon}, method="linear")

    u_ageo = maps.uo.values - u_geo_xr_int.values[0]
    v_ageo = maps.vo.values - v_geo_xr_int.values[0]

    
    # Créer un DataArray pour "u"
    u_ageo_xr = xr.DataArray(
        u_ageo,
        dims=("lat", "lon"),
        coords={
            "lat": lat_u[:,0],
            "lon": lon_u[0,:],
        },
        name="uageo"
    ).expand_dims(time=[current_date])

    # Créer un DataArray pour "u"
    v_ageo_xr = xr.DataArray(
        v_ageo,
        dims=("lat", "lon"),
        coords={

            "lat": lat_u[:,0],
            "lon": lon_u[0,:],
        },
        name="vageo"
    ).expand_dims(time=[current_date])
    

    #print("interp 4th ...")
    u_geo_4th = u_geo_xr.interp({"lat":lat_4th, "lon":lon_4th}, method="linear")
    v_geo_4th = v_geo_xr.interp({"lat":lat_4th, "lon":lon_4th}, method="linear")
    u_ageo_4th = u_ageo_xr.interp({"lat":lat_4th, "lon":lon_4th}, method="linear")
    v_ageo_4th = v_ageo_xr.interp({"lat":lat_4th, "lon":lon_4th}, method="linear")

    # Ajouter à la liste
    ugeo_list.append(u_geo_4th)
    vgeo_list.append(v_geo_4th)
    uageo_list.append(u_ageo_4th)
    vageo_list.append(v_ageo_4th)
    time_list.append(current_date)

    current_date += timedelta(days=1)
    time_index=time_index+1
    pbar.update(1)
    # Nettoyer le terminal
    #clear_output(wait=True)


# Fermer la barre de progression
pbar.close()

# Concaténer tous les DataArrays le long de la dimension "time"
ugeo = xr.concat(ugeo_list, dim="time")
vgeo = xr.concat(vgeo_list, dim="time")
uageo = xr.concat(uageo_list, dim="time")
vageo = xr.concat(vageo_list, dim="time")
# Créer le Dataset final
ds = xr.Dataset({"ugeo": ugeo, "vgeo": vgeo,"uageo": uageo, "vageo": vageo})


print("Saving...")
save_file=f"/Odyssey/private/t22picar/data/glorys_{depth}/glorys_uv_geo_and_ageo_{depth}_2010-2018.nc"

# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)