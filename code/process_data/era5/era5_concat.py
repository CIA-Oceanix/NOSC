import xarray as xr

path_era="/Odyssey/private/t22picar/data/era5/"
# Charger les deux fichiers NetCDF
ds1 = xr.open_dataset(path_era+"era5_2010-01-01_2020-01-01_8th.nc").sel(time=slice("2010-01-01","2019-12-31"))
ds2 = xr.open_dataset(path_era+"era5_2020-01-01_2021-01-01_8th.nc").sel(time=slice("2020-01-01","2021-01-01"))

# Fusionner les deux jeux de données le long de la dimension 'time'
ds_fusionne = xr.concat([ds1, ds2], dim="time")

# Sauvegarder le résultat dans un nouveau fichier
ds_fusionne.to_netcdf(path_era+"era5_2010-01-01_2021-01-01_8th.nc")