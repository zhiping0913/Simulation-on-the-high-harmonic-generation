import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.scipy.special import erf 
import numpy as np
import scipy.constants as C
import os
from joblib import Parallel, delayed
from scipy.ndimage import rotate, map_coordinates
import pandas as pd
import xarray as xr







def write_field_3D(Field_list:list[jnp.ndarray],x_axis:jnp.ndarray,y_axis:jnp.ndarray,z_axis:jnp.ndarray,name_list=list[str],nc_name='',working_dir=''):
    x_axis=jnp.asarray(x_axis,dtype=jnp.float64)
    y_axis=jnp.asarray(y_axis,dtype=jnp.float64)
    z_axis=jnp.asarray(z_axis,dtype=jnp.float64)
    Nx=x_axis.size
    Ny=y_axis.size
    Nz=z_axis.size
    assert len(Field_list)==len(name_list)
    data_vars={}
    for Field,name in zip(Field_list,name_list):
        Field=jnp.asarray(Field,dtype=jnp.float64)
        assert Field.shape==(Nx,Ny,Nz), f"Field shape {Field.shape} does not match expected shape {(Nx,Ny,Nz)}."
        data_vars[name]=(["x", "y", "z"], Field)
    field_ds=xr.Dataset(
        data_vars=data_vars,
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis),'z':(["z"], z_axis)}
        )
    nc_path=os.path.join(working_dir,nc_name)
    field_ds.to_netcdf(path=nc_path,format="NETCDF4", engine='h5netcdf')
    print('Write field to %s' %nc_path)
    return nc_path


