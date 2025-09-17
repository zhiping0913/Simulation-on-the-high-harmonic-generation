import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.ndimage import rotate,zoom,shift,map_coordinates
from scipy.integrate import simpson
from scipy.signal import peak_widths, find_peaks,hilbert
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
import pandas as pd
import cv2
import xarray as xr
from scipy.special import erf 
from start import read_sdf,read_nc,read_dat

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/no_curve/45/2D/ND_a0=1.00'

def continue_field_2D(Field:np.ndarray,n_continuation_x:int,n_continuation_y:int,center_shift=(0,0),edge_length=100):
    """
        Extend the field array to a shape (n_continuation_x,n_continuation_y) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_length: int. the length (number of cells) of the smoothing area at the edge
    """
    n_x,n_y=Field.shape
    assert n_x<=n_continuation_x-2*np.abs(center_shift[0])
    assert n_y<=n_continuation_y-2*np.abs(center_shift[1])
    grid_center_mask=np.s_[round((n_continuation_x-n_x)/2)+center_shift[0]:round((n_continuation_x-n_x)/2)+n_x+center_shift[0],round((n_continuation_y-n_y)/2)+center_shift[1]:round((n_continuation_y-n_y)/2)+n_y+center_shift[1]]
    x_id,y_id=np.meshgrid(np.arange(n_x),np.arange(n_y),indexing='ij')
    x_left_trans=0.5 * (1 + erf((x_id - edge_length) /edge_length))
    x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-edge_length)) / edge_length))
    y_left_trans=0.5 * (1 + erf((y_id - edge_length) /edge_length))
    y_right_trans= 0.5 * (1 - erf((y_id - (n_y-1-edge_length)) / edge_length))   #smooth the edge
    Field_continuation=np.zeros(shape=(n_continuation_x,n_continuation_y))
    Field_continuation[grid_center_mask]=Field*x_left_trans*x_right_trans*y_left_trans*y_right_trans
    return Field_continuation



def rotate_perpendicular_field_2D(Field_z:np.ndarray,angle=0.0):
    """
    The direction of the field is perpendicular to the plane of rotation (x-y plane).
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_z (np.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.
    """
    n_x,n_y=Field_z.shape
    Field_z_rotate=rotate(input=Field_z,angle=np.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_z_rotate_continuation=continue_field_2D(Field=Field_z_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_z_rotate_continuation

def rotate_xy(Field_x:np.ndarray,Field_y:np.ndarray,angle=0.0):
    """_summary_
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_x (np.ndarray): _description_
        Field_y (np.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    Returns:
        _type_: _description_
    """
    assert Field_x.shape==Field_y.shape
    n_x,n_y=Field_x.shape
    Field_y_1=Field_x*np.sin(angle)+Field_y*np.cos(angle)
    Field_x_1=Field_x*np.cos(angle)-Field_y*np.sin(angle)
    Field_y_1_rotate=rotate(input=Field_y_1,angle=np.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_x_1_rotate=rotate(input=Field_x_1,angle=np.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_y_1_rotate_continuation=continue_field_2D(Field=Field_y_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    Field_x_1_rotate_continuation=continue_field_2D(Field=Field_x_1_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_x_1_rotate_continuation,Field_y_1_rotate_continuation


def rotate_parallel_field_2D(Field_x:np.ndarray,Field_y:np.ndarray,x_axis:np.ndarray,y_axis:np.ndarray,angle=0.0):
    """
    The direction of the field is in the plane of rotation (x-y plane).
    Rotate the field to change the direction of propagation to +x direction. 
    Transversal component
    longitudinal component
    Args:
        Field_x (np.ndarray): _description_
        Field_y (np.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    """
    assert x_axis.ndim==1
    assert y_axis.ndim==1
    n_x=x_axis.size
    n_y=y_axis.size
    assert Field_x.shape==(n_x,n_y)
    assert Field_y.shape==(n_x,n_y)
    x,y=np.meshgrid(x_axis,y_axis,indexing='ij')
    d_x=x_axis[1]-x_axis[0]
    d_y=y_axis[1]-y_axis[0]
    Field_x_rotate_continuation,Field_y_rotate_continuation=rotate_xy(Field_x,Field_y,angle)
    x_rotate_continuation,y_rotate_continuation=rotate_xy(x,y,angle)
    Field_y_rotate_max=np.max(np.abs(Field_y_rotate_continuation))
    Field_y_rotate_max_id=tuple(np.array(np.where(np.abs(Field_y_rotate_continuation)==Field_y_rotate_max))[:,0])
    Field_center_mask=np.s_[round(Field_y_rotate_max_id[0]-n_x/2):round(Field_y_rotate_max_id[0]+n_x/2),round(Field_y_rotate_max_id[1]-n_y/2):round(Field_y_rotate_max_id[1]+n_y/2)]
    print('Input field shape: ',Field_y.shape)
    print('transversal_rotate_max_id: ',Field_y_rotate_max_id)
    Field_y_rotate_max_x=x_rotate_continuation[Field_y_rotate_max_id]
    Field_y_rotate_max_y=y_rotate_continuation[Field_y_rotate_max_id]
    x_axis_rotate=np.linspace(start=Field_y_rotate_max_x-d_x*n_x/2,stop=Field_y_rotate_max_x+d_x*n_x/2,num=n_x,endpoint=False)
    y_axis_rotate=np.linspace(start=Field_y_rotate_max_y-d_y*n_y/2,stop=Field_y_rotate_max_y+d_y*n_y/2,num=n_y,endpoint=False)
    Field_x_rotate=Field_x_rotate_continuation[Field_center_mask]
    Field_y_rotate=Field_y_rotate_continuation[Field_center_mask]
    print('Square integrate (should be close to 1.0)')
    print((np.einsum('ij,ij->',Field_x_rotate,Field_x_rotate)+np.einsum('ij,ij->',Field_y_rotate,Field_y_rotate))/(np.einsum('ij,ij->',Field_x,Field_x)+np.einsum('ij,ij->',Field_y,Field_y)))
    return {
        'Field_x_rotate': Field_x_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate':Field_y_rotate,   #shape=(n_x,n_y)
        'Field_y_rotate_max_id':Field_y_rotate_max_id,   #the id of max of transversal_rotate in (2*n_x,2*n_y) grid
        'Field_center_mask':Field_center_mask,   #
        'x_axis_rotate':x_axis_rotate,   #shape=(n_x,)
        'y_axis_rotate':y_axis_rotate,   #shape=(n_y,)
    }

def shift_field_2D(Field:np.ndarray,x_axis_0:np.ndarray,y_axis_0:np.ndarray,x_axis_1:np.ndarray,y_axis_1:np.ndarray):
    """
    Shift the field from original grid to new grid. 
    Example: Shift the field from the boundary of the grid to the center of the grid.
    Example: Zoom the field with a different resolution.
    Example: Choose fields within our area of interest.
    Args:
        Field (np.ndarray): _description_
        x_axis_0 (np.ndarray): 1D array. Original x grid.
        y_axis_0 (np.ndarray): 1D array. Original y grid.
        x_axis_1 (np.ndarray): 1D array. New x grid.
        y_axis_1 (np.ndarray): 1D array. New y grid.
    """
    assert Field.ndim==2
    assert x_axis_0.ndim==1
    assert y_axis_0.ndim==1
    assert x_axis_1.ndim==1
    assert y_axis_1.ndim==1
    assert Field.shape==(x_axis_0.size,y_axis_0.size)
    print('Original shape')
    print(Field.shape)
    d_x_0=x_axis_0[1]-x_axis_0[0]
    d_y_0=y_axis_0[1]-y_axis_0[0]
    d_x_1=x_axis_1[1]-x_axis_1[0]
    d_y_1=y_axis_1[1]-y_axis_1[0]
    x_axis_1_in_0_id=(x_axis_1 - x_axis_0[0]) / d_x_0   #the position (id,could be non-integer) of new x in original x
    y_axis_1_in_0_id=(y_axis_1 - y_axis_0[0]) / d_y_0
    x_1_in_0_id,y_1_in_0_id=np.meshgrid(x_axis_1_in_0_id,y_axis_1_in_0_id,indexing='ij')
    Field_1=map_coordinates(input=Field, coordinates=np.array([x_1_in_0_id,y_1_in_0_id]), order=3, mode='nearest')
    """    
    Field_interpolator=RegularGridInterpolator(
        points=(x_axis_0,y_axis_0),
        values=Field,
        method="cubic",
        bounds_error=False,
        fill_value=0,
        )
    x_1,y_1=np.meshgrid(x_axis_1,y_axis_1,indexing='ij')
    Field_1=Field_interpolator((x_1,y_1))
    """
    print('New shape')
    print(Field_1.shape)
    print('Square integrate (should be close to 1.0)')
    print((np.einsum('ij,ij->',Field_1,Field_1)*d_x_1*d_y_1)/(np.einsum('ij,ij->',Field,Field)*d_x_0*d_y_0))
    return Field_1

def smooth_edge_2D(Field:np.ndarray,mask:np.ndarray,edge_length=100):
    assert Field.shape==mask.shape
    n_x,n_y=Field.shape
    mask=np.ones(shape=(n_x,n_y))*mask
    x_min_id=np.full(shape=(n_y),fill_value=np.nan)
    x_max_id=np.full(shape=(n_y),fill_value=np.nan)
    for y_id in range(n_y):
        x_id_nonzero=np.nonzero(mask[:,y_id])[0]
        if x_id_nonzero.size>0:
            x_min_id[y_id]=x_id_nonzero[0]
            x_max_id[y_id]=x_id_nonzero[-1]
    y_min_id=np.full(shape=(n_x),fill_value=np.nan)
    y_max_id=np.full(shape=(n_x),fill_value=np.nan)
    for x_id in range(n_x):
        y_id_nonzero=np.nonzero(mask[x_id,:])[0]
        if y_id_nonzero.size>0:
            y_min_id[x_id]=y_id_nonzero[0]
            y_max_id[x_id]=y_id_nonzero[-1]
    x_id_axis=np.arange(n_x,dtype=np.int64)
    y_id_axis=np.arange(n_y,dtype=np.int64)
    xx_id,yy_id=np.meshgrid(x_id_axis,y_id_axis,indexing='ij')
    x_left_trans=0.5 * (1 + erf((xx_id - x_min_id.reshape(1, -1)-edge_length) /edge_length))
    x_right_trans= 0.5 * (1 - erf((xx_id - x_max_id.reshape(1, -1)+edge_length) / edge_length))
    y_left_trans=0.5 * (1 + erf((yy_id - y_min_id.reshape(-1, 1)-edge_length) /edge_length))
    y_right_trans= 0.5 * (1 - erf((yy_id -y_max_id.reshape(-1, 1)+edge_length) / edge_length)) 
    mask=np.nan_to_num(mask*x_left_trans*x_right_trans*y_left_trans*y_right_trans,nan=0)
    return Field*mask

    
def write_field_2D(Field_list:list[np.ndarray],x_axis:np.ndarray,y_axis:np.ndarray,name_list=list[str],nc_name='',working_dir=''):
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
    field_ds.to_netcdf(path=os.path.join(working_dir,nc_name),format="NETCDF4", engine='h5netcdf')
    print(nc_name)
    return 0

if __name__ == "__main__": 


    i=1

    #data_dict_B=read_sdf(sdf_name=os.path.join(working_dir,'%0.4d.sdf' %(i)),block_name_list=['Magnetic_Field_Bz'])
    data_dict=read_nc(nc_name=os.path.join(working_dir,'0001_500cpl.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'])
    #data_dict=read_sdf(sdf_name=os.path.join(working_dir,'0001.sdf'),block_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'])

    laser_lambda = 0.8*C.micron		# Laser wavelength
    vacuum_length_x_lambda=18
    vacuum_length_y_lambda=18
    cells_per_lambda_x=500
    cells_per_lambda_y=500
    cells_per_lambda_new=500
    n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
    n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda_y)
    d_x=laser_lambda/cells_per_lambda_x
    d_y=laser_lambda/cells_per_lambda_y


    x_min=-vacuum_length_x_lambda*laser_lambda
    x_max=vacuum_length_x_lambda*laser_lambda
    y_min=-vacuum_length_y_lambda*laser_lambda
    y_max=vacuum_length_y_lambda*laser_lambda

    x_axis=np.linspace(start=x_min,stop=x_max,num=n_field_x,endpoint=False)+d_x/2
    y_axis=np.linspace(start=y_min,stop=y_max,num=n_field_y,endpoint=False)+d_y/2
    xb_axis=x_axis+d_x/2
    yb_axis=y_axis+d_y/2

    n_field_x_new=round(2*vacuum_length_x_lambda*cells_per_lambda_new)
    d_x_new=laser_lambda/cells_per_lambda_new
    x_axis_new=np.linspace(start=x_min,stop=x_max,num=n_field_x_new,endpoint=False)+d_x_new/2
    y_axis_new=x_axis_new

    x,y=np.meshgrid(x_axis_new,y_axis_new,indexing='ij')

    Electric_Field_Ex_1=data_dict['Electric_Field_Ex'][:n_field_x//2,n_field_y//2:]
    Electric_Field_Ey_1=data_dict['Electric_Field_Ey'][:n_field_x//2,n_field_y//2:]
    Magnetic_Field_Bz_1=data_dict['Magnetic_Field_Bz'][:n_field_x//2,n_field_y//2:]


    #Electric_Field_Ex_1=shift_field_2D(Field=Electric_Field_Ex,x_axis_0=xb_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Electric_Field_Ey_1=shift_field_2D(Field=Electric_Field_Ey,x_axis_0=x_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
    #Magnetic_Field_Bz_1=shift_field_2D(Field=Magnetic_Field_Bz,x_axis_0=xb_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)




    Electric_Field_Ex_1=smooth_edge_2D(Electric_Field_Ex_1,mask=np.ones_like(Electric_Field_Ex_1),edge_length=2*cells_per_lambda_new)
    Electric_Field_Ey_1=smooth_edge_2D(Electric_Field_Ey_1,mask=np.ones_like(Electric_Field_Ey_1),edge_length=2*cells_per_lambda_new)
    Magnetic_Field_Bz_1=smooth_edge_2D(Magnetic_Field_Bz_1,mask=np.ones_like(Magnetic_Field_Bz_1),edge_length=2*cells_per_lambda_new)
    rotate_parallel_field_2D_dict=rotate_parallel_field_2D(Field_x=Electric_Field_Ex_1,Field_y=Electric_Field_Ey_1,x_axis=x_axis_new[:n_field_x_new//2],y_axis=y_axis_new[n_field_x_new//2:],angle=-3*np.pi/4)
    Electric_Field_Ex_rotate=rotate_parallel_field_2D_dict['Field_x_rotate']
    Electric_Field_Ey_rotate=rotate_parallel_field_2D_dict['Field_y_rotate']
    Magnetic_Field_Bz_rotate=rotate_perpendicular_field_2D(Magnetic_Field_Bz_1,angle=-3*np.pi/4)[rotate_parallel_field_2D_dict['Field_center_mask']]
    x_axis_rotate=rotate_parallel_field_2D_dict['x_axis_rotate']
    y_axis_rotate=rotate_parallel_field_2D_dict['y_axis_rotate']



    write_field_2D(Field_list=[Electric_Field_Ex_rotate,Electric_Field_Ey_rotate,Magnetic_Field_Bz_rotate],
                x_axis=x_axis_rotate,y_axis=y_axis_rotate,
                name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],nc_name=os.path.join(working_dir,'0001_reflection_rotate_500cpl.nc'))

    exit(0)
"transversal"
"longitudinal"
"incident"
'reflection,reflected'
'transmission,transmitted'