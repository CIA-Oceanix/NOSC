import warnings
warnings.filterwarnings("ignore")
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from glob import glob
import sys
sys.path.append("/Odyssey/private/t22picar/tools")
from jaxparrow import cyclogeostrophy, geostrophy
from tqdm import tqdm  # Importer tqdm


save_file=f"/Odyssey/public/UNET_rec_sla/IMT_uvgos_2010-01-01_2020-01-01_4th.nc"

ssh_imt = "/Odyssey/public/UNET_rec_sla/mapping/Unet_SSH_2010-01-01_2020-01-01_4th.nc"
ssh_imt = xr.open_dataset(ssh_imt)

depth="15m"
file_data = f"/Odyssey/private/t22picar/data/glorys_{depth}/glorys_multivar_{depth}_2010-2018.nc"
maps_4th = xr.open_dataset(file_data)
#maps=maps.sel(time="2010-01-01")
lon_ref = maps_4th.lon.values
lat_ref = maps_4th.lat.values

# Get longitudes and latitudes
lon = ssh_imt.lon.values
lat = ssh_imt.lat.values
lon2D, lat2D = np.meshgrid(lon, lat)

start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 1, 1)
current_date = start_date
time_index=0
ugeo_list=[]
vgeo_list=[]
time_list=[]

# Initialiser la barre de progression
total_iterations = ssh_imt.time.values.shape[0]
pbar = tqdm(total=total_iterations, desc="Ajout des pas de temps")

while current_date < end_date:
    #print(current_date)
    ssh_imt
    map = ssh_imt.sel(time=current_date)
    #Compute geostrophy
    (u_geo,v_geo,lat_u, lon_u, lat_v, lon_v) = geostrophy(map.zos.values, lat2D, lon2D)

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

    #print("interp 4th ...")
    u_geo_xr = u_geo_xr.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
    v_geo_xr = v_geo_xr.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")

    # Ajouter à la liste
    ugeo_list.append(u_geo_xr)
    vgeo_list.append(v_geo_xr)
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
# Créer le Dataset final
ds = xr.Dataset({"ugos": ugeo, "vgos": vgeo})


print("Saving...")

# Sauvegarder le DataArray en fichier NetCDF
ds.to_netcdf(save_file)