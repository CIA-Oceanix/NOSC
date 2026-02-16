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

FMT = '%Y-%m-%dT%H:%M:%S.%fZ'

#ifile = "/Odyssey/private/t22picar/data/drifters/Drifters_CMEMS_GL_15m_20100101T000000Z_20200101T000000Z.pyo"
#with open(ifile, 'rb') as f:
#    dic_pyo = pickle.load(f)


def make_nc_file(ofile,dvar,odic,ATTR: Optional[dict] = {}):

    if os.path.isfile(ofile):
        os.remove(ofile)
    handler = netCDF4.Dataset(ofile, 'w')
    listkey = list(odic.keys())
    #dvargcp = f'{dvar}_gcp'
    dvargcp = dvar
    #dim = handler.createDimension(dvar, len(odic[dvar]))
    dimgcp = handler.createDimension(dvargcp, len(odic[dvar]))
    var = handler.createVariable("index_time_gcp", 'i4', (dvargcp))
    var.long_name = "index of ground control points in time dimension" ;
    var.comment = "index goes from 0 (start of first pixel) to dimension value (end of last pixel)" ;
    var[:] = numpy.arange(len(odic[dvar]))

    for key in listkey:
        
        _tmp = numpy.array(odic[key])

        if (type(_tmp[0]) != datetime.datetime) and (type(_tmp[0]) != str):
            _tmp = numpy.array(odic[key], dtype='float32')
        
        if (type(_tmp[0]) != str): #(_tmp.dtype == float) or (type(_tmp[0]) == datetime.datetime):
            #print(type(_tmp[0]))
            _type = 'f4'
            
            if "lon" in key:
                var = handler.createVariable("lon", 'f4', (dvargcp,))
                var.long_name = "ground control points longitude"
                var.standard_name = "longitude"
                var.units = "degrees_east"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp

            elif "lat" in key:
                var = handler.createVariable("lat", 'f4', (dvargcp,))
                var.long_name = "ground control points latitude"
                var.standard_name = "latitude"
                var.units = "degrees_north"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp

            elif "date" in key:
                var = handler.createVariable("date", 'f8', (dvargcp,))
                var.long_name = "date"
                var.standard_name = "date"
                var.units = "seconds since 1970-01-01T00:00:00.000000Z"
                var.calendar = "standard"
                var.axis = "T"

            elif "ums" in key:
                var = handler.createVariable("ums", 'f8', (dvargcp,))
                var.long_name = "West-east current component at the drog depht"
                var.standard_name = "ums"
                var.units = "m s-1"
                if _tmp.shape[1] != 1:
                    _tmp = _tmp[:,:1]
                var[:] = _tmp

            elif "vms" in key:
                var = handler.createVariable("vms", 'f8', (dvargcp,))
                var.long_name = "South-north current component at the drog depht"
                var.standard_name = "vms"
                var.units = "m s-1"
                if _tmp.shape[1] != 1:
                    _tmp = _tmp[:,:1]
                var[:] = _tmp
                
            """
            elif "time" in key:
                var = handler.createVariable("time", 'f8', (dvargcp,))
                var.long_name = "time"
                var.standard_name = "time"
                var.units = "seconds since 1970-01-01T00:00:00.000000Z"
                var.calendar = "standard"
                var.axis = "T"
            """
            """
            else:
                var = handler.createVariable(key, 'f4', (dvar),
                                                fill_value=1e36)
                #_tmp, scale, offset = pack_as_ubytes(_tmp, numpy.nanmin(_tmp), numpy.nanmax(_tmp))
                #var.scale_factor = scale
                #var.add_offset = offset
                #var.long_name = key
                #var.unit = ""
                #var.valid_min = numpy.ubyte(0) ;
                #var.valid_max = numpy.ubyte(254) ;
                #var[:] = _tmp
                var.scale_factor = 1.0
                var.add_offset = 0.0
                var.long_name = key
                var.units = ""
                var.valid_min = -1000.0 ;
                var.valid_max = 1000.0 ;
                var[:] = _tmp
            """
            if key in ATTR.keys():
                if 'unit' in ATTR[key].keys():
                    var.units = ATTR[key]['unit']
                if 'long_name' in ATTR[key].keys():
                    var.long_name = ATTR[key]['long_name']
                if 'valid_min' in ATTR[key].keys():
                    var.valid_min = ATTR[key]['valid_min']     
                if 'valid_max' in ATTR[key].keys():
                    var.valid_max = ATTR[key]['valid_max']  
                if 'description' in ATTR[key].keys():
                    var.comment = ATTR[key]['description']    
            if _tmp.dtype == float:
                var[:] = _tmp
            elif type(_tmp[0]) == datetime.datetime:
                ref = datetime.datetime(1970, 1, 1)
                _tmps = [(_t - ref).total_seconds() for _t in _tmp]
                var[:] = _tmps
                var.units = 'seconds since 1970-01-01T00:00:00.000000Z'
                start = _tmp[0].strftime(FMT)
                stop = _tmp[-1].strftime(FMT)
        else:
            print(key)
    handler.cdm_data_type = "Trajectory"
    handler.idf_version = "1.0"
    handler.idf_granule_id = os.path.basename(ofile)
    handler.time_coverage_start = start
    handler.time_coverage_end = stop
    handler.idf_subsampling_factor = 0
    handler.idf_spatial_resolution = 1.e+07
    handler.idf_spatial_resolution_units = "m"
    handler.processing_software = ""
    handler.netcdf_version_id = "4.7.4 of Oct 31 2021 02:46:37 $"
    handler.standard_name_vocabulary = "NetCDF Climate and Forecast (CF) Metadata Convention"
    handler.close()


### Add year files

#for year in range(2010,2025):
for year in range(2010,2024):
    print(year)
    ifile = f"/Odyssey/private/t22picar/data/drifters/drifter_00m/CMEMS_GL_00/Drifters_CMEMS_GL_00m_{year}0101T000000Z_{year+1}0101T000000Z.pyo.gz"
    with gzip.open(ifile, 'rb') as f:
        dic_pyo = pickle.load(f)

    dvar="date"
    for ide, odic in dic_pyo.items():
        for i in range(len(odic["date"])):
            odic["date"][i] = datetime.datetime.strptime(str(odic["date"][i]),'%Y-%m-%d %H:%M:%S')
        ide=ide.replace(" ", "")
        #odic["date"][:] = pd.to_datetime(odic["date"][:])
        ofile = os.path.join('/Odyssey/public/drifters/cmems_00m/', f'drifter_{ide}_{year}.nc')

        make_nc_file(ofile,dvar,odic)
###


"""
dvar="date"
for ide, odic in dic_pyo.items():
    for i in range(len(odic["date"])):
        odic["date"][i] = datetime.datetime.strptime(str(odic["date"][i]),'%Y-%m-%d %H:%M:%S')
    ide=ide.replace(" ", "")
    #odic["date"][:] = pd.to_datetime(odic["date"][:])
    ofile = os.path.join('/Odyssey/private/t22picar/data/drifters/cmems/', f'drifter_{ide}.nc')

    make_nc_file(ofile,dvar,odic)
"""