import xarray as xr
import numpy as np
#ds = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2019/gridded_0.125deg/sla_l3_all_2010_2019_0.125deg_convl4_float32.nc', chunks = ({'time': 365, 'latitude': 1000, 'longitude': 1500}))
ds = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2019/gridded_0.125deg/sla_l3_all_2010_2019_0.125deg_convl4.nc', chunks = ({'time': 365, 'latitude': 1000, 'longitude': 1500}))

#ds['time'] = ds_old['time']

#print('Saving started')
#ds.to_netcdf('compressed_sla_l3_time_test.nc', encoding={'latitude': {'zlib': False},\
#longitude': {'zlib': False},\
#'time': {'zlib': False},\
#'sla': {'chunksizes': [365,1000,1000], 'zlib': True,\
#'complevel': 1}})



# Convertir les variables 'time' et 'sla' en float32
ds['sla'] = ds['sla'].astype('float32')
ds['longitude'] = ds['longitude'].astype(np.float32)
ds['latitude'] = ds['latitude'].astype(np.float32) # attributes
#s['time'] = ds['time'].astype(np.int32)   # atributes

COMPLEVEL = 4
ZLIB = True
FV32 = 1.e+20
ENC_FV = {'zlib': ZLIB, 'complevel': COMPLEVEL, '_FillValue': FV32,
          'dtype': 'float32'}
ENC = {'zlib': ZLIB, 'complevel': COMPLEVEL, 'dtype': 'float32'}
ENC_T = {'zlib': ZLIB, 'complevel': COMPLEVEL}


print('Saving started')
ds.to_netcdf('compressed_sla_l3_complevelLucile.nc', encoding={'latitude': ENC,\
'longitude': ENC,\
'time': ENC_T,\
'sla': ENC_FV})

