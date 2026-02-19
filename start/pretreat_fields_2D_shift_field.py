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
from start import read_sdf,read_nc,read_dat

def square_integral_field_2D(Field:jnp.ndarray,d_x=1,d_y=1,complex_array=False):
    assert Field.ndim==2
    if complex_array:
        Field=jnp.asarray(Field,dtype=jnp.complex128)
        square_integral=jnp.real(jnp.einsum('ij,ij->',Field,jnp.conjugate(Field)))*d_x*d_y
    else:
        Field=jnp.asarray(Field,dtype=jnp.float64)
        square_integral=jnp.einsum('ij,ij->',Field,Field)*d_x*d_y
    square_integral=float(square_integral)
    print('∬|Field|^2×dx×dy=%e' %(square_integral))
    return square_integral
#@profile
def continue_field_2D(Field:jnp.ndarray,n_continuation_x:int,n_continuation_y:int,center_shift=(0,0),smooth=True,edge_length=100):
    """
        Extend the field array to a shape (n_continuation_x,n_continuation_y) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_length: int. the length (number of cells) of the smoothing area at the edge
    """
    n_x,n_y=Field.shape
    assert n_x<=n_continuation_x-2*jnp.abs(center_shift[0])
    assert n_y<=n_continuation_y-2*jnp.abs(center_shift[1])
    grid_center_mask=jnp.s_[round((n_continuation_x-n_x)/2)+center_shift[0]:round((n_continuation_x-n_x)/2)+n_x+center_shift[0],round((n_continuation_y-n_y)/2)+center_shift[1]:round((n_continuation_y-n_y)/2)+n_y+center_shift[1]]
    Field_continuation=jnp.zeros(shape=(n_continuation_x,n_continuation_y))
    if smooth:
        x_id=jnp.arange(n_x)
        y_id=jnp.arange(n_y)
        x_left_trans=0.5 * (1 + erf((x_id - edge_length) /edge_length))
        x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-edge_length)) / edge_length))
        y_left_trans=0.5 * (1 + erf((y_id - edge_length) /edge_length))
        y_right_trans= 0.5 * (1 - erf((y_id - (n_y-1-edge_length)) / edge_length))   #smooth the edge
        Field_continuation=Field_continuation.at[grid_center_mask].set(jnp.einsum('ij,i,i,j,j->ij',Field,x_left_trans,x_right_trans,y_left_trans,y_right_trans))
    else:
        Field_continuation=Field_continuation.at[grid_center_mask].set(Field)
    return Field_continuation



def rotate_perpendicular_field_2D(Field_z:jnp.ndarray,angle=0.0):
    """
    The direction of the field is perpendicular to the plane of rotation (x-y plane).
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_z (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.
    """
    n_x,n_y=Field_z.shape
    Field_z_rotate=rotate(input=Field_z,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_z_rotate_continuation=continue_field_2D(Field=Field_z_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_z_rotate_continuation

def rotate_xy(Field_x:jnp.ndarray,Field_y:jnp.ndarray,angle=0.0):
    """_summary_
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_x (jnp.ndarray): _description_
        Field_y (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    Returns:
        _type_: _description_
    """
    assert Field_x.shape==Field_y.shape
    n_x,n_y=Field_x.shape
    Field_y_1=Field_x*jnp.sin(angle)+Field_y*jnp.cos(angle)
    Field_x_1=Field_x*jnp.cos(angle)-Field_y*jnp.sin(angle)
    Field_y_1_rotate=rotate(input=Field_y_1,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_x_1_rotate=rotate(input=Field_x_1,angle=jnp.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_y_1_rotate_continuation=continue_field_2D(Field=Field_y_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    Field_x_1_rotate_continuation=continue_field_2D(Field=Field_x_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_x_1_rotate_continuation,Field_y_1_rotate_continuation


def rotate_parallel_field_2D(Field_x:jnp.ndarray,Field_y:jnp.ndarray,x_axis:jnp.ndarray,y_axis:jnp.ndarray,angle=0.0):
    """
    The direction of the field is in the plane of rotation (x-y plane).
    Rotate the field to change the direction of propagation to +x direction. 
    Transversal component
    longitudinal component
    Args:
        Field_x (jnp.ndarray): _description_
        Field_y (jnp.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    """
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    n_x=x_axis.size
    n_y=y_axis.size
    assert Field_x.shape==(n_x,n_y)
    assert Field_y.shape==(n_x,n_y)
    x,y=jnp.meshgrid(x_axis,y_axis,indexing='ij')
    d_x=x_axis[1]-x_axis[0]
    d_y=y_axis[1]-y_axis[0]
    Field_x_rotate_continuation,Field_y_rotate_continuation=rotate_xy(Field_x,Field_y,angle)
    x_rotate_continuation,y_rotate_continuation=rotate_xy(x,y,angle)
    Field_y_rotate_max=jnp.max(jnp.abs(Field_y_rotate_continuation))
    Field_y_rotate_max_id=tuple(jnp.array(jnp.where(jnp.abs(Field_y_rotate_continuation)==Field_y_rotate_max))[:,0])
    Field_center_mask=jnp.s_[round(Field_y_rotate_max_id[0]-n_x/2):round(Field_y_rotate_max_id[0]+n_x/2),round(Field_y_rotate_max_id[1]-n_y/2):round(Field_y_rotate_max_id[1]+n_y/2)]
    print('Input field shape: ',Field_y.shape)
    print('transversal_rotate_max_id: ',Field_y_rotate_max_id)
    Field_y_rotate_max_x=x_rotate_continuation[Field_y_rotate_max_id]
    Field_y_rotate_max_y=y_rotate_continuation[Field_y_rotate_max_id]
    x_axis_rotate=jnp.linspace(start=Field_y_rotate_max_x-d_x*n_x/2,stop=Field_y_rotate_max_x+d_x*n_x/2,num=n_x,endpoint=False)
    y_axis_rotate=jnp.linspace(start=Field_y_rotate_max_y-d_y*n_y/2,stop=Field_y_rotate_max_y+d_y*n_y/2,num=n_y,endpoint=False)
    Field_x_rotate=Field_x_rotate_continuation[Field_center_mask]
    Field_y_rotate=Field_y_rotate_continuation[Field_center_mask]
    print('Square integrate (should be close to 1.0)')
    print((square_integral_field_2D(Field_x_rotate)+square_integral_field_2D(Field_y_rotate))/(square_integral_field_2D(Field_x)+square_integral_field_2D(Field_y)))
    return {
        'Field_x_rotate': Field_x_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate':Field_y_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate_max_id':Field_y_rotate_max_id,   #the id of max of transversal_rotate in (2*n_x,2*n_y) grid
        'Field_center_mask':Field_center_mask,   #
        'x_axis_rotate':x_axis_rotate,   #shape=(n_x,)
        'y_axis_rotate':y_axis_rotate,   #shape=(n_y,)
    }

def shift_field_2D(Field:jnp.ndarray,x_axis_0:jnp.ndarray,y_axis_0:jnp.ndarray,x_axis_1:jnp.ndarray,y_axis_1:jnp.ndarray):
    """
    Shift the field from original grid to new grid. 
    Example: Shift the field from the boundary of the grid to the center of the grid.
    Example: Zoom the field with a different resolution.
    Example: Choose fields within our area of interest.
    Args:
        Field (jnp.ndarray): _description_
        x_axis_0 (jnp.ndarray): 1D array. Original x grid.
        y_axis_0 (jnp.ndarray): 1D array. Original y grid.
        x_axis_1 (jnp.ndarray): 1D array. New x grid.
        y_axis_1 (jnp.ndarray): 1D array. New y grid.
    """
    assert Field.ndim==2
    assert x_axis_0.ndim==1
    assert y_axis_0.ndim==1
    assert x_axis_1.ndim==1
    assert y_axis_1.ndim==1
    assert Field.shape==(x_axis_0.size,y_axis_0.size)
    print('Original shape')
    print(Field.shape)
    d_x_0=(x_axis_0[-1]-x_axis_0[0])/(x_axis_0.size-1)
    d_y_0=(y_axis_0[-1]-y_axis_0[0])/(y_axis_0.size-1)
    d_x_1=(x_axis_1[-1]-x_axis_1[0])/(x_axis_1.size-1)
    d_y_1=(y_axis_1[-1]-y_axis_1[0])/(y_axis_1.size-1)
    x_axis_1_in_0_id=(x_axis_1 - x_axis_0[0]) / d_x_0   #the position (id,could be non-integer) of new x in original x
    y_axis_1_in_0_id=(y_axis_1 - y_axis_0[0]) / d_y_0
    x_1_in_0_id,y_1_in_0_id=jnp.meshgrid(x_axis_1_in_0_id,y_axis_1_in_0_id,indexing='ij')
    Field_1=map_coordinates(input=Field, coordinates=jnp.array([x_1_in_0_id,y_1_in_0_id]), order=5, mode='constant', cval=0.0)
    print('New shape')
    print(Field_1.shape)
    print('Square integrate (should be close to 1.0)')
    print(square_integral_field_2D(Field=Field_1,d_x=d_x_1,d_y=d_y_1)/square_integral_field_2D(Field=Field,d_x=d_x_0,d_y=d_y_0))
    return Field_1

def smooth_edge_2D(Field:jnp.ndarray,mask:jnp.ndarray,edge_length=100):
    n_x,n_y=Field.shape
    assert mask.shape==(n_x,n_y)
    threshold=1e-3
    x_id=jnp.arange(n_x)
    y_id=jnp.arange(n_y)
    mask=jnp.ones(shape=(n_x,n_y),dtype=jnp.int8)*mask
    sum_x=jnp.sum(mask,axis=0)   #shape=(n_y,)
    sum_y=jnp.sum(mask,axis=1)   #shape=(n_x,)
    x_min_id=jnp.where(sum_x<threshold, 2*n_x-1, jnp.argmax(mask,axis=0))   #shape=(n_y,)
    x_max_id=jnp.where(sum_x<threshold, -n_x, n_x-1-jnp.argmax(mask[::-1,:],axis=0))   #shape=(n_y,)
    y_min_id=jnp.where(sum_y<threshold, 2*n_y-1, jnp.argmax(mask,axis=1))   #shape=(n_x,)
    y_max_id=jnp.where(sum_y<threshold, -n_y, n_y-1-jnp.argmax(mask[:,::-1],axis=1))   #shape=(n_x,)
    x_left_trans=0.5 * (1 + erf((x_id[:,jnp.newaxis] - x_min_id[jnp.newaxis,:]-edge_length) /edge_length))   #shape==(n_x,n_y)
    x_right_trans= 0.5 * (1 - erf((x_id[:,jnp.newaxis] - x_max_id[jnp.newaxis,:]+edge_length) / edge_length))   #shape==(n_x,n_y)
    y_left_trans=0.5 * (1 + erf((y_id[jnp.newaxis,:] - y_min_id[:,jnp.newaxis]-edge_length) /edge_length))   #shape==(n_x,n_y)
    y_right_trans= 0.5 * (1 - erf((y_id[jnp.newaxis,:] -y_max_id[:,jnp.newaxis]+edge_length) / edge_length))    #shape==(n_x,n_y)
    mask_smooth=x_left_trans+x_right_trans+y_left_trans+y_right_trans-3
    mask_smooth=mask_smooth*(mask_smooth>threshold)
    Field_smooth=Field*mask_smooth
    print(square_integral_field_2D(Field_smooth)/square_integral_field_2D(Field))
    return Field_smooth

    
def write_field_2D(Field_list:list[jnp.ndarray],x_axis:jnp.ndarray,y_axis:jnp.ndarray,name_list=list[str],nc_name='',working_dir=''):
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    assert len(Field_list)==len(name_list)
    data_vars={}
    for Field,name in zip(Field_list,name_list):
        Field=np.asarray(Field,dtype=np.float64)
        assert Field.shape==(x_axis.size,y_axis.size)
        #Field.transpose(1,0).tofile(os.path.join(working_dir,name))
        data_vars[name]=(["x", "y"], Field)
    field_ds=xr.Dataset(
        data_vars=data_vars,
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,nc_name),format="NETCDF4", engine='h5netcdf')
    print(os.path.join(working_dir,nc_name))
    return 0

def write_field(field:jnp.ndarray,x_axis:jnp.ndarray,y_axis:jnp.ndarray,name=''):
    np.asarray(field.transpose(1,0),dtype=np.float64).tofile(os.path.join(working_dir,name))
    print(os.path.join(working_dir,name))
    return 0
    field_ds=xr.Dataset(
        data_vars={
            name:(["x", "y"], field),
            },
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)),format="NETCDF4", engine='h5netcdf')
    print(os.path.join(working_dir,'%s.nc' %(name)))
    return 0



working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=20/2D/Initialize_Field'
print(working_dir)

data_dict=read_nc(nc_name=os.path.join(working_dir,'Field0000.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'])
Electric_Field_Ex=data_dict['Electric_Field_Ex']
Electric_Field_Ey=data_dict['Electric_Field_Ey']
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
x_axis=data_dict['x']
y_axis=data_dict['y']

laser_lambda = 0.8*C.micron		# Laser wavelength
vacuum_length_x_lambda=50
vacuum_length_y_lambda=50
cells_per_lambda_x=100
cells_per_lambda_y=100
cells_per_lambda_x_new=1000
cells_per_lambda_y_new=500
x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda
n_field_x_new=round(2*vacuum_length_x_lambda*cells_per_lambda_x_new)
n_field_y_new=round(2*vacuum_length_y_lambda*cells_per_lambda_y_new)
d_x_new=laser_lambda/cells_per_lambda_x_new
d_y_new=laser_lambda/cells_per_lambda_y_new
x_axis_new=jnp.linspace(start=x_min,stop=x_max,num=n_field_x_new,endpoint=False,dtype=jnp.float64)+d_x_new/2
y_axis_new=jnp.linspace(start=y_min,stop=y_max,num=n_field_y_new,endpoint=False,dtype=jnp.float64)+d_y_new/2
xb_axis_new=x_axis_new+d_x_new/2
yb_axis_new=y_axis_new+d_y_new/2

Electric_Field_Ex_1=shift_field_2D(Field=Electric_Field_Ex,x_axis_0=data_dict['x'],y_axis_0=data_dict['y'],x_axis_1=xb_axis_new,y_axis_1=y_axis_new)
Electric_Field_Ey_1=shift_field_2D(Field=Electric_Field_Ey,x_axis_0=data_dict['x'],y_axis_0=data_dict['y'],x_axis_1=x_axis_new,y_axis_1=yb_axis_new)
Magnetic_Field_Bz_1=shift_field_2D(Field=Magnetic_Field_Bz,x_axis_0=data_dict['x'],y_axis_0=data_dict['y'],x_axis_1=xb_axis_new,y_axis_1=yb_axis_new)

def write_field(field:jnp.ndarray,x_axis=x_axis,y_axis=y_axis,name=''):
    np.asarray(field.transpose(1,0),dtype=np.float64).tofile(os.path.join(working_dir,name))
    print(os.path.join(working_dir,name))
    return 0
write_field(field=Electric_Field_Ex_1,x_axis=xb_axis_new,y_axis=y_axis_new,name='Electric_Field_Ex.dat')
write_field(field=Electric_Field_Ey_1,x_axis=x_axis_new,y_axis=yb_axis_new,name='Electric_Field_Ey.dat')
write_field(field=Magnetic_Field_Bz_1,x_axis=xb_axis_new,y_axis=yb_axis_new,name='Magnetic_Field_Bz.dat')