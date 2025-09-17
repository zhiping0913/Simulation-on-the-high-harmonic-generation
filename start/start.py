import scipy.constants as C
import math
import os
import subprocess
import numpy as np
import xarray as xr
import sdf_helper
from scipy.special import erf 

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
    print('Read %s' %(dat_name))
    print(shape)
    data=np.fromfile(dat_name,dtype=np.float64)
    data=data.reshape(np.flip(shape))
    data=data.transpose()
    return data



def smooth_edge(Field:np.ndarray,mask:np.ndarray,edge_length=100):
    """
    The edge of the field array is reduced to 0 to avoid the noice at the edge.
    Args:
        Field (np.ndarray): _description_
        mask (np.ndarray): _description_
        edge_length: The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_length: int. the length (number of cells) of the smoothing area at the edge.

    Returns:
        _type_: _description_
    """
    assert Field.shape==mask.shape
    dim=Field.ndim
    shape=np.array(Field.shape,dtype=np.int64)
    mask=np.ones(shape=shape)*mask
    mask_nonzero_ids=np.array(np.nonzero(mask))   #shape=(dim,mask_nonzero_ids_num)
    mask_nonzero_ids_num=mask_nonzero_ids.shape[1]
    min_id_list=[]
    max_id_list=[]
    trans_list=[]
    for dim_i in range(dim):
        dim_i_min_ids=np.full(shape=np.delete(arr=shape,obj=dim_i),fill_value=np.nan)   #eg. When dim_i=0, dim_i_min_ids.shape=(shape[1],shape[2],…)
        dim_i_max_ids=np.full(shape=np.delete(arr=shape,obj=dim_i),fill_value=np.nan)
        min_id_list.append(dim_i_min_ids)
        max_id_list.append(dim_i_max_ids)
    if mask_nonzero_ids_num>0:
        for mask_nonzero_id_i in range(mask_nonzero_ids_num):
            mask_nonzero_id=mask_nonzero_ids[:,mask_nonzero_id_i]
            for dim_i in range(dim):
                dim_i_mask_nonzero_id=tuple(np.delete(arr=mask_nonzero_id,obj=dim_i))   #shape=(dim-1,)
                dim_i_min_nonzero_id=min_id_list[dim_i][dim_i_mask_nonzero_id]
                dim_i_max_nonzero_id=max_id_list[dim_i][dim_i_mask_nonzero_id]
                if np.isnan(dim_i_min_nonzero_id) or dim_i_min_nonzero_id>mask_nonzero_id[dim_i]:
                    min_id_list[dim_i][dim_i_mask_nonzero_id]=mask_nonzero_id[dim_i]
                if np.isnan(dim_i_max_nonzero_id) or dim_i_max_nonzero_id<mask_nonzero_id[dim_i]:
                    max_id_list[dim_i][dim_i_mask_nonzero_id]=mask_nonzero_id[dim_i]
    for dim_i in range(dim):
        dim_i_id_axis=np.arange(stop=shape[dim_i])
        dim_i_id_axis_expand=np.expand_dims(a=dim_i_id_axis,axis=tuple(np.delete(arr=np.arange(dim),obj=dim_i)))   #eg. When dim_i=1, dim_i_id_axis_expand.shape=(1,shape[1],1,…)
        dim_i_min_ids_expand=np.expand_dims(a=min_id_list[dim_i],axis=dim_i)   #eg. When dim_i=1, dim_i_min_ids_expand.shape=(shape[0],1,shape[2],…)
        dim_i_max_ids_expand=np.expand_dims(a=max_id_list[dim_i],axis=dim_i)
        dim_i_min_trans=0.5 * (1 + erf((dim_i_id_axis_expand - dim_i_min_ids_expand-edge_length) /edge_length))   #dim_i_min_trans.shape=shape
        dim_i_max_trans=0.5 * (1 - erf((dim_i_id_axis_expand - dim_i_max_ids_expand+edge_length) / edge_length))
        dim_i_trans=dim_i_min_trans*dim_i_max_trans
        trans_list.append(dim_i_trans)
    trans=np.multiply.reduce(array=np.array(trans_list),axis=0)
    mask_out=np.nan_to_num(mask*trans,nan=0)
    return Field*mask_out


