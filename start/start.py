import scipy.constants as C
import math
import os
import subprocess
import numpy as np
import xarray as xr
import sdf_helper


class block():
    def __init__(self,built:str):
        self.built=built
    
    def print(self):
        output='begin : %s\n' %(self.__dict__['built'])
        for key in self.__dict__.keys():
            if key=='built':
                continue
            elif key=='supplemental':
                output=output+'\t%s\n' %(self.__dict__['supplemental'])
            else:
                output=output+"\t%s=%s\n" %(key,self.__dict__[key])
        output=output+'end : %s\n\n' %(self.__dict__['built'])
        print(output)
        return output

def read_sdf(sdf_name='',block_name_list=[]):
    sdf=sdf_helper.getdata(fname=sdf_name)
    data_dict={}
    for block_name in block_name_list:
        data_dict[block_name]=sdf.__dict__[block_name].data
    return data_dict

def read_nc(nc_name='',key_name_list=[]):
    print('Read %s' %(nc_name))
    nc=xr.open_dataset(filename_or_obj=nc_name)
    data_dict={}
    for key_name in key_name_list:
        data=nc[key_name].to_numpy()
        data_dict[key_name]=data
        print(key_name)
        print(data.shape)
    return data_dict

def read_dat(dat_name='',shape=()):
    data=np.fromfile(dat_name,dtype=np.float64)
    data=data.reshape(np.flip(shape))
    data=data.transpose()
    return data

def write_field_2D(Field_list:list[np.ndarray],x_axis:np.ndarray,y_axis:np.ndarray,name_list=list[str],nc_name=''):
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    assert len(Field_list)==len(name_list)
    data_vars={}
    for Field,name in zip(Field_list,name_list):
        assert Field.shape==(x_axis.size,y_axis.size)
        #Field.transpose(1,0).tofile(os.path.join(working_dir,name))
        data_vars[name]=(["x", "y"], Field)
    field_ds=xr.Dataset(
        data_vars=data_vars,
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=nc_name)
    print(nc_name)
    return 0

