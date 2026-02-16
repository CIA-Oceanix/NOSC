import numpy as np
import xarray as xr 
import os

def regrid_da(da: xr.DataArray,lon_ref,lat_ref):
        new_da = da.interp({"lat":lat_ref, "lon":lon_ref}, method="linear")
        return new_da


def interpolation_glorys_raw_data(year_start,year_end,depth,folder_data,lon_ref,lat_ref):

    print(year_start)
    folder_glorys = f"/Odyssey/private/t22picar/data/glorys_{depth}m/"
    file_glorys = f"glorys_multivar_{depth}m_{year_start}.nc"
    maps_glorys = xr.open_dataset(folder_glorys+file_glorys).sel(time=slice(f"{year_start}-01-01",f"{year_start}-12-31"))
    maps_glorys = maps_glorys.sel(depth=maps_glorys.depth[0]).drop_vars("mlotst")

    if depth==15:
        folder_glorys_0m = f"/Odyssey/private/t22picar/data/glorys_0m/"
        file_glorys_0m = f"glorys_multivar_0m_{year_start}.nc"
        maps_glorys_0m = xr.open_dataset(folder_glorys_0m+file_glorys_0m).sel(time=slice(f"{year_start}-01-01",f"{year_start}-12-31"))
        maps_glorys_0m = maps_glorys_0m.sel(depth=maps_glorys_0m.depth[0]).thetao       
        maps_glorys["thetao"] =  maps_glorys_0m

    # Get the list of variable names
    variable_names = list(maps_glorys.variables.keys())
    variable_names.remove("time")
    for var in variable_names:
        maps_glorys[var] = maps_glorys[var].astype(np.float32)
        
    print("Interpolation glorys ... ")
    maps_glorys = maps_glorys.rename({"latitude": "lat"})
    maps_glorys = maps_glorys.rename({"longitude": "lon"})
    maps_glorys = regrid_da(maps_glorys,lon_ref,lat_ref)
    print("Interpolation done ")



    for year in range(year_start+1,year_end+1):
        print(year)
        file_glorys = f"glorys_multivar_{depth}m_{year}.nc"
        maps_glorys_i = xr.open_dataset(folder_glorys+file_glorys).sel(time=slice(f"{year}-01-01",f"{year}-12-31",))
        maps_glorys_i = maps_glorys_i.sel(depth=maps_glorys_i.depth[0]).drop_vars("mlotst")

        if depth==15:
            folder_glorys_0m = f"/Odyssey/private/t22picar/data/glorys_0m/"
            file_glorys_0m = f"glorys_multivar_0m_{year_start}.nc"
            maps_glorys_0m = xr.open_dataset(folder_glorys_0m+file_glorys_0m).sel(time=slice(f"{year_start}-01-01",f"{year_start}-12-31"))
            maps_glorys_0m = maps_glorys_0m.sel(depth=maps_glorys_0m.depth[0]).thetao       
            maps_glorys_i["thetao"] =  maps_glorys_0m

        # Get the list of variable names
        variable_names = list(maps_glorys_i.variables.keys())
        variable_names.remove("time")

        for var in variable_names:
            maps_glorys_i[var] = maps_glorys_i[var].astype(np.float32)

        print("Interpolation glorys... ")
        maps_glorys_i = maps_glorys_i.rename({"latitude": "lat"})
        maps_glorys_i = maps_glorys_i.rename({"longitude": "lon"})  
        maps_glorys_i = regrid_da(maps_glorys_i,lon_ref,lat_ref)
        print("Interpolation glorys done ")
        print("Concatenation ... ")
        maps_glorys = xr.concat([maps_glorys, maps_glorys_i], dim='time')
        print("Concatenation done ")

    #maps_glorys = maps_glorys.sel(depth=maps_glorys.depth[0])

    # save data 
    print("Saving...")
    save_file=f"glorys_multivar_{year_start}_{year_end}.nc"

    # Sauvegarder le DataArray en fichier NetCDF
    maps_glorys.to_netcdf(folder_data+save_file)
    print("Saving done")

def interpolation_era5_raw_data(year_start,year_end,folder_data,lon_ref,lat_ref,size_grid):

    print("interpolation era5")
    print(year_start)

    folder_era5 = "/Odyssey/private/t22picar/data/era5/"
    file_era5 = f"era5_{year_start}.grib" # Actually not glorys 
    map_era5 = xr.open_dataset(folder_era5+file_era5, engine="cfgrib").sel(time=slice(f"{year_start}-01-01",f"{year_start}-12-31"))
    map_era5['longitude'] = xr.where(map_era5['longitude'] > 180, map_era5['longitude'] - 360, map_era5['longitude'])
    map_era5 = map_era5.rename({"latitude": "lat"})
    map_era5 = map_era5.rename({"longitude": "lon"})

    # Daily mean wind
    print("computing daily mean era5 ...")
    map_era5 = map_era5.resample(valid_time='1D').mean()
    print("computation daily mean done")
    # Interpolation new grid
    print("Interpolating era5 ...")
    map_era5 = regrid_da(map_era5,lon_ref,lat_ref)
    print("interpolation done")
    #Split into two files
    map_era5 = map_era5.drop("number").drop("step").drop("surface").rename({"valid_time": "time"})

    # Get the list of variable names
    variable_names = list(map_era5.variables.keys())
    variable_names.remove("time")

    for var in variable_names:
        map_era5[var] = map_era5[var].astype(np.float32)

    for year in range(year_start+1,year_end+1):
    
        file_era5 = f"era5_{year}.grib" # Actually not glorys 
        map_era5_i = xr.open_dataset(folder_era5+file_era5, engine="cfgrib").sel(time=slice(f"{year}-01-01",f"{year}-12-31"))
        map_era5_i['longitude'] = xr.where(map_era5_i['longitude'] > 180, map_era5_i['longitude'] - 360, map_era5_i['longitude'])
        map_era5_i = map_era5_i.rename({"latitude": "lat"})
        map_era5_i = map_era5_i.rename({"longitude": "lon"})

        # Daily mean wind
        print("computing daily mean era5 ...")
        map_era5_i = map_era5_i.resample(valid_time='1D').mean()
        print("computation daily mean done")
        # Interpolation new grid
        print("Interpolating era5 ...")
        map_era5_i = regrid_da(map_era5_i,lon_ref,lat_ref)
        print("interpolation done")
        #Split into two files
        map_era5_i = map_era5_i.drop("number").drop("step").drop("surface").rename({"valid_time": "time"})

        # Get the list of variable names
        variable_names = list(map_era5_i.variables.keys())
        variable_names.remove("time")

        map_era5 = xr.concat([map_era5, map_era5_i], dim='time')

    # save data
    print("Saving file ...")
    save_file=f"era5_{year_start}_{year_end}_{size_grid}.nc"

    # Sauvegarder le DataArray en fichier NetCDF
    map_era5.to_netcdf(folder_data+save_file)
    print("Saving done")



def interpolation_ssh_raw_data(year,depth,folder_data,lon_ref,lat_ref): 

    time_end = "2019-12-31"
    time_start = "2019-01-01"
    folder_ssh = "/Odyssey/private/t22picar/data/ssh_L4/"
    file_obs = "SSH_L4_CMEMS_2019.nc" # Actually not glorys 

    maps = xr.open_dataset(folder_ssh+file_obs).sel(time=slice(time_start,time_end))
    maps = maps.rename({"latitude": "lat"})
    maps = maps.rename({"longitude": "lon"})
    maps = maps.rename({"adt": "zos"})

    # Get the list of variable names
    variable_names = list(maps.variables.keys())
    variable_names.remove("time")

    for var in variable_names:
        maps[var] = maps[var].astype(np.float32)

    folder_glorys = f"/Odyssey/private/t22picar/data/glorys_{depth}m/"
    file_glorys = f"glorys_multivar_{depth}m_2019.nc"
    maps_glo = xr.open_dataset(folder_glorys+file_glorys).sel(time=slice(time_start,time_end))

    mean_glorys_ssh = np.nanmean(maps_glo.zos.values)
    mean_sat_ssh = np.nanmean(maps.zos.values)
    offset = mean_glorys_ssh-mean_sat_ssh

    maps.zos.values = maps.zos.values + offset

    print("Interpolation cmems ssh ... ")
    maps = regrid_da(maps,lon_ref,lat_ref)
    print("Interpolation done")

    # save data 
    print("Saving...")
    save_file=folder_data+f"cmems_ssh_{year}.nc"
    print(save_file)
    # Sauvegarder le DataArray en fichier NetCDF
    maps.to_netcdf(save_file)
    print("Saving done")

# SST 

def interpolation_sst_raw_data(year,depth,folder_data,lon_ref,lat_ref):


    time_end = "2019-12-31"
    time_start = "2019-01-01"
    folder_sst = "/Odyssey/private/t22picar/data/sst_L4/"
    file_sst = "SST_L4_OSTIA_2019.nc"

    maps = xr.open_dataset(folder_sst+file_sst).sel(time=slice(time_start,time_end))
    maps = maps.rename({"latitude": "lat"})
    maps = maps.rename({"longitude": "lon"})
    maps = maps.rename({"analysed_sst": "thetao"})

    # Get the list of variable names
    variable_names = list(maps.variables.keys())
    variable_names.remove("time")

    for var in variable_names:
        maps[var] = maps[var].astype(np.float32)


    print("Interpolation ostia ssh ... ")
    # Interpolation new grid
    maps = regrid_da(maps,lon_ref,lat_ref)
    maps.thetao.values = maps.thetao.values - 273.15
    print("Interpolation done")


    # save data 
    print("Saving...")
    save_file=folder_data+f"ostia_sst_{year}"+".nc"
    print(save_file)
    # Sauvegarder le DataArray en fichier NetCDF
    maps.to_netcdf(save_file)
    print("Saving done")

def interpolation_era5_raw_data_test(folder_data,lon_ref,lat_ref):

    time_end = "2019-12-31"
    time_start = "2019-01-01"
    print("open data")
    folder_era5 = "/Odyssey/private/t22picar/data/era5/"
    file_era5 = "era5_2019.grib" # Actually not glorys 
    map_era5 = xr.open_dataset(folder_era5+file_era5, engine="cfgrib").sel(time=slice(time_start,time_end))
    map_era5['longitude'] = xr.where(map_era5['longitude'] > 180, map_era5['longitude'] - 360, map_era5['longitude'])
    map_era5 = map_era5.rename({"latitude": "lat"})
    map_era5 = map_era5.rename({"longitude": "lon"})

    # Daily mean wind
    print("computing daily mean era5 ...")
    map_era5 = map_era5.resample(valid_time='1D').mean()
    print("computation daily mean done")
    # Interpolation new grid
    print("Interpolating era5 ...")
    map_era5 = regrid_da(map_era5,lon_ref,lat_ref)
    print("interpolation done")
    #Split into two files
    map_era5 = map_era5.drop("number").drop("step").drop("surface").rename({"valid_time": "time"})

    # Get the list of variable names
    variable_names = list(map_era5.variables.keys())
    variable_names.remove("time")

    for var in variable_names:
        map_era5[var] = map_era5[var].astype(np.float32)
    # save data
    print("Saving file ...")
    save_file=f"era5_2019.nc"

    # Sauvegarder le DataArray en fichier NetCDF
    map_era5.to_netcdf(folder_data+save_file)
    print("Saving done")
