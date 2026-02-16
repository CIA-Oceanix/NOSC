# vim: ts=4:sts=4:sw=4
# @author Lucile Gaultier
# @date 2025-04-30


import numpy
import netCDF4
import datetime
import os
from typing import Optional, Tuple

FMT = '%Y-%m-%dT%H:%M:%S.%fZ'

def write_selection_csv(ofile: str, odic: dict):
    odic.pop('Time')
    listkey = list(odic.keys())
    size = len(odic[listkey[0]])
    with open(ofile, 'wt') as f:
        f.write(','.join(listkey))
        f.write('\n')
        for i in range(size):
            ilist = [str(odic[key][i]) for key in listkey]
            line = ','.join(ilist)
            f.write(line)
            f.write('\n')


def make_config(listkey, dic_pickle, ifile, label=None):
    if label is None:
        label = os.path.splitext(os.path.basename(ifile))[0]
    for key in listkey:
        if "array" in dic_pickle[key].keys():
            
            #try:
            if True:
                _min = numpy.nanmin(dic_pickle[key]["array"])#[~dic_pickle[key]["array"].mask])
                _max = numpy.nanmax(dic_pickle[key]["array"])#[~dic_pickle[key]["array"].mask])
                if _min == numpy.nan:
                    _min = 0
                    _max = 1
            #except:
            else:
                _min = 0
                _max = 1
        if "min" not in dic_pickle[key].keys():
            dic_pickle[key]["min"] =_min
        if "max" not in dic_pickle[key].keys():
            dic_pickle[key]["max"] =_max

    with open(ifile, 'w') as f:
        f.write('[general]\n')
        f.write(f'label = {label}\n')
        # Ryyf.write('NEWSAligned = false\n')
        f.write('xSeamless = false\n')
        f.write('ySeamless = false\n')
        f.write('mustBeCurrent = false\n')
        f.write('tags = type:insitu\n')
        f.write(f'defaultVariable = {listkey[5]}\n')
        listname = [value["name"] for k,value in dic_pickle.items()]
        strlist = ','.join(listname)
        f.write(f'variables = {strlist}\n')
        f.write('\n')
        for key in listkey:
            f.write(f'[{dic_pickle[key]["name"]}]\n')
            f.write(f'label = {key} \n')
            f.write(f'fields = {dic_pickle[key]["name"]}\n')
            f.write(f'units = {dic_pickle[key]["unit"]}\n')
            f.write('defaultRendering = TRAJECTORIES\n')
            f.write('logscale = false\n')
            f.write(f'min = {dic_pickle[key]["min"]}\n')
            f.write(f'max = {dic_pickle[key]["max"]}\n')
            f.write('opacity = 0.5\n')
            f.write('zindex = 0.1\n')
            f.write('particlesCount = 0\n')
            f.write('particleTTL = 0\n')
            f.write('streamlineLength = 0\n')
            f.write('streamlineSpeed = 0.0\n')
            f.write('colormap = medspiration\n')
            f.write('color = 0,0,0\n')
            f.write('tags = parameter:insitu\n')
            f.write('filterMode = BILINEAR\n')
            f.write('\n')

def pack_as_ubytes(var: numpy.ndarray, vmin: float, vmax: float
                  )-> Tuple[numpy.ndarray, float, float]:
    nan_mask = ((numpy.ma.getmaskarray(var)) | (numpy.isnan(var))
                )

    offset, scale = vmin, (float(vmax) - float(vmin)) / 254.0
    if vmin == vmax:
        scale = 1.0

    numpy.clip(var, vmin, vmax, out=var)

    # Required to avoid runtime warnings on masked arrays wherein the
    # division of the _FillValue by the scale cannot be stored by the dtype
    # of the array
    if isinstance(var, numpy.ma.MaskedArray):
        mask = numpy.ma.getmaskarray(var).copy()
        var[numpy.where(mask)] = vmin
        _var = (numpy.ma.getdata(var) - offset) / scale
        var.mask = mask  # Restore mask to avoid side-effects
    else:
        _var = (var - offset) / scale

    result = numpy.round(_var).astype('ubyte')
    result[numpy.where(nan_mask)] = 255
    return result, scale, offset


def write_netcdf_idf(ofile:str, odic: dict, dvar: str,
                     ATTR: Optional[dict] = {}
                     ):
    if os.path.isfile(ofile):
        os.remove(ofile)
    handler = netCDF4.Dataset(ofile, 'w')
    listkey = list(odic.keys())
    dvargcp = f'{dvar}_gcp'
    dim = handler.createDimension(dvar, len(odic[dvar]))
    dimgcp = handler.createDimension(dvargcp, len(odic[dvar]))
    var = handler.createVariable("index_time_gcp", 'i4', (dvargcp))
    var.long_name = "index of ground control points in time dimension" ;
    var.comment = "index goes from 0 (start of first pixel) to dimension value (end of last pixel)" ;

    var[:] = numpy.arange(len(odic[dvar]))
    for key in listkey:
        _tmp = numpy.array(odic[key])
        #if not odic[key] is list:
        #    continue
            #setattr(handler, key, _tmp)
        if (type(_tmp[0]) != datetime.datetime) and (type(_tmp[0]) != str):
            _tmp = numpy.array(odic[key], dtype='float32')
        
        if (type(_tmp[0]) != str): #(_tmp.dtype == float) or (type(_tmp[0]) == datetime.datetime):
            _type = 'f4'
            
            if "lon" in key:
                var = handler.createVariable("lon_gcp", 'f4', (dvargcp,))
                var.long_name = "ground control points longitude"
                var.standard_name = "longitude"
                var.units = "degrees_east"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp

            elif "lat" in key:
                var = handler.createVariable("lat_gcp", 'f4', (dvargcp,))
                var.long_name = "ground control points latitude"
                var.standard_name = "latitude"
                var.units = "degrees_north"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp
            elif "time" in key:
                var = handler.createVariable("time", 'f8', (dvargcp,))
                var.long_name = "time"
                var.standard_name = "time"
                var.units = "seconds since 1970-01-01T00:00:00.000000Z"
                var.calendar = "standard"
                var.axis = "T"
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
    

def write_netcdf(ofile:str, odic: dict, dvar: str,
                 ATTR: Optional[dict] = {}
                 ):
    if os.path.isfile(ofile):
        os.remove(ofile)
    handler = netCDF4.Dataset(ofile, 'w')
    listkey = list(odic.keys())
    #dvargcp = f'{dvar}_gcp'
    dim = handler.createDimension(dvar, len(odic[dvar]))
    #dimgcp = handler.createDimension(dvargcp, len(odic[dvar]))
    #var = handler.createVariable("index_time_gcp", 'i4', (dvargcp))
    #var.long_name = "index of ground control points in time dimension" ;
    #var.comment = "index goes from 0 (start of first pixel) to dimension value (end of last pixel)" ;

    #var[:] = numpy.arange(len(odic[dvar]))
    for key in listkey:
        _tmp = numpy.array(odic[key])
        #if not odic[key] is list:
        #    continue
            #setattr(handler, key, _tmp)
        if (type(_tmp[0]) != datetime.datetime) and (type(_tmp[0]) != str):
            _tmp = numpy.array(odic[key], dtype='float32')
        
        if (type(_tmp[0]) != str): #(_tmp.dtype == float) or (type(_tmp[0]) == datetime.datetime):
            _type = 'f4'
            
            if "lon" in key:
                var = handler.createVariable("lon", 'f4', (dvar,))
                var.long_name = "longitude"
                var.standard_name = "longitude"
                var.units = "degrees_east"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp

            elif "lat" in key:
                var = handler.createVariable("lat", 'f4', (dvar,))
                var.long_name = "latitude"
                var.standard_name = "latitude"
                var.units = "degrees_north"
                var.comment = "geographical coordinates, WGS84 projection"
                var[:] = _tmp
            elif "time" in key:
                var = handler.createVariable("time", 'f8', (dvar,))
                var.long_name = "time"
                var.standard_name = "time"
                var.units = "seconds since 1970-01-01T00:00:00.000000Z"
                var.calendar = "standard"
                var.axis = "T"
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
