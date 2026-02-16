import sys   
import xarray as xr
import numpy as np

# # Récupération du xp_name 
path_ncfile = sys.argv[1]
#path_ncfile = "/Odyssey/private/t22picar/data/ssh_L4/SSH_L4_CMEMS_2010-01-01-2019-01-01_4th.nc" #sys.argv[1]
ds = xr.open_dataset(path_ncfile)

# Get the list of variable names
variable_names = list(ds.variables.keys())
variable_names.remove("time")

for var in variable_names:
    ds[var] = ds[var].astype(np.float32)

ds.to_netcdf(path_ncfile[:-3]+"_float32.nc")