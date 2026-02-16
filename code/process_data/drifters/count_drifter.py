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
from datetime import timedelta, datetime
import os
from typing import Optional, Tuple

type_drifter = "cmems_00m"

for year in range(2010,2022):
    print(year)
    filenames_drifters = sorted(glob(f'/Odyssey/public/drifters/{type_drifter}/drifter_*_{year}.nc'))
    ds_drifter = xr.open_mfdataset(filenames_drifters, combine='nested', concat_dim='date')
    year_ds_drifter = ds_drifter.sel(date=ds_drifter["date.year"] == year)
    year_ds_drifter = year_ds_drifter.compute()
    year_ds_drifter_filtred = year_ds_drifter.where(year_ds_drifter['ums'] < 1000, drop=True)

    count = []
    start_date = datetime(year, 1, 1)
    #end_date = datetime(2019, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date=start_date
    while current_date < end_date:
        count.append(len(list(year_ds_drifter.sel(date = current_date.strftime("%Y-%m-%d")).date)))
        current_date += timedelta(days=1)  # Ajoute 1 an (approximation)

    plt.figure(figsize=(8, 4))  # Taille de la figure (largeur, hauteur)
    plt.plot(count)
    plt.xlabel(f"days in {year}")
    plt.ylabel("nb data")


    plt.savefig(f"./plot/{type_drifter}/drifters_count_{year}.png")