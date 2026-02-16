import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd 
import sys 

folder_data = sys.argv[1]
print("file to modify")
print(folder_data)

def add_pad(folder_data):
    ds_map = xr.open_dataset(folder_data)
    lon_space = np.linspace(-180,179.75,1440)
    ds_map = ds_map.interp({"lon":lon_space}, method="linear")
    ds_map = ds_map.pad(lon=4, mode="wrap")
    # Mettre à jour les coordonnées de longitude pour refléter le nouveau domaine
    ds_map['lon'] = np.linspace(-181, 180.75, len(ds_map['lon']))
    return ds_map

ds_map_padded = add_pad(folder_data)

save_file=folder_data[:-3]+"_pad.nc"
# Sauvegarder le DataArray en fichier NetCDF
print("Saving ...")
ds_map_padded.to_netcdf(save_file)
print("Saving done")
