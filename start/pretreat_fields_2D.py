import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.ndimage import rotate,zoom
from scipy.integrate import simpson
from scipy.signal import peak_widths, find_peaks,hilbert
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
import pandas as pd
import cv2
import xarray as xr
from scipy.special import erf 
from start import read_sdf,read_nc,read_dat,write_field_2D

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thin/2D/rotate_field'

def continue_field(Field:np.ndarray,n_continuation_x:int,n_continuation_y:int,edge_length=100):
    """
        Extend the field array to a shape (n_continuation_x,n_continuation_y) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_length: int. the length (number of cells) of the smoothing area at the edge
    """
    n_x,n_y=Field.shape
    assert n_x<=n_continuation_x
    assert n_y<=n_continuation_y
    grid_center_mask=np.s_[round((n_continuation_x-n_x)/2):round((n_continuation_x-n_x)/2)+n_x,round((n_continuation_y-n_y)/2):round((n_continuation_y-n_y)/2)+n_y]
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
    Field_z_rotate_continuation=continue_field(Field=Field_z_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    return Field_z_rotate_continuation
    
    




def rotate_parallel_field_2D(Field_x:np.ndarray,Field_y:np.ndarray,angle=0.0):
    """
    The direction of the field is in the plane of rotation (x-y plane).
    Rotate the field to change the direction of propagation to +x direction. 
    Transversal component
    longitudinal component
    Note that: The output fields' shape is twice the input fields'. output.shape=2*input.shape=(2*input.shape[0],2*input.shape[1])
    Args:
        Field_x (np.ndarray): _description_
        Field_y (np.ndarray): _description_
        angle: The angle from the wave's direction to +x direction.Unit: rad. Negative angle means that the wave vector is in the first or second quadrant.

    """
    assert Field_x.shape==Field_y.shape
    n_x,n_y=Field_y.shape
    Field_transversal=Field_x*np.sin(angle)+Field_y*np.cos(angle)
    Field_longitudinal=Field_x*np.cos(angle)-Field_y*np.sin(angle)
    Field_transversal_rotate=rotate(input=Field_transversal,angle=np.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_longitudinal_rotate=rotate(input=Field_longitudinal,angle=np.rad2deg(angle),axes=(0,1),reshape=True,mode='constant',cval=0,order=3)
    Field_transversal_rotate_continuation=continue_field(Field=Field_transversal_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    Field_longitudinal_rotate_continuation=continue_field(Field=Field_longitudinal_rotate,n_continuation_x=2*n_x,n_continuation_y=2*n_y,edge_length=max(n_x,n_y)//20)
    Field_transversal_rotate_max=np.max(np.abs(Field_transversal_rotate))
    Field_transversal_rotate_max_id=tuple(np.array(np.where(np.abs(Field_transversal_rotate_continuation)==Field_transversal_rotate_max))[:,0])
    Field_center_mask=np.s_[round(Field_transversal_rotate_max_id[0]-n_x/2):round(Field_transversal_rotate_max_id[0]+n_x/2),round(Field_transversal_rotate_max_id[1]-n_y/2):round(Field_transversal_rotate_max_id[1]+n_y/2)]
    print('Input field shape: ',Field_y.shape)
    print('transversal_rotate_max_id: ',Field_transversal_rotate_max_id)
    return Field_longitudinal_rotate_continuation,Field_transversal_rotate_continuation,Field_center_mask

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
    Field_interpolator=RegularGridInterpolator(
        points=(x_axis_0,y_axis_0),
        values=Field,
        method="cubic",
        bounds_error=False,
        fill_value=0,
        )
    x_1,y_1=np.meshgrid(x_axis_1,y_axis_1,indexing='ij')
    Field_1=Field_interpolator((x_1,y_1))
    print('New shape')
    print(Field_1.shape)
    return Field_1

def smooth_edge(Field:np.ndarray,mask:np.ndarray,edge_length=100):
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
    

i=23
#data_dict_B=read_sdf(sdf_name=os.path.join(working_dir,'%0.4d.sdf' %(i)),block_name_list=['Magnetic_Field_Bz'])
data_dict=read_nc(nc_name=os.path.join(working_dir,'Field_%0.4d_250cpl.nc' %(i)),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'])

laser_lambda = 0.8*C.micron		# Laser wavelength
vacuum_length_x_lambda=16
vacuum_length_y_lambda=16
cells_per_lambda_x=250
cells_per_lambda_y=250
cells_per_lambda_new=250
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

Electric_Field_Ex=data_dict['Electric_Field_Ex']
Electric_Field_Ey=data_dict['Electric_Field_Ey']
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
#Electric_Field_Ex_1=shift_field_2D(Field=Electric_Field_Ex,x_axis_0=xb_axis,y_axis_0=y_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
#Electric_Field_Ey_1=shift_field_2D(Field=Electric_Field_Ey,x_axis_0=x_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)
#Magnetic_Field_Bz_1=shift_field_2D(Field=Magnetic_Field_Bz,x_axis_0=xb_axis,y_axis_0=yb_axis,x_axis_1=x_axis_new,y_axis_1=y_axis_new)

reflection_mask=(x>0)*(y>0)
Electric_Field_Ex_reflection=smooth_edge(Electric_Field_Ex,reflection_mask,edge_length=2*cells_per_lambda_new)[n_field_x//2:,n_field_y//2:]
Electric_Field_Ey_reflection=smooth_edge(Electric_Field_Ey,reflection_mask,edge_length=2*cells_per_lambda_new)[n_field_x//2:,n_field_y//2:]
Magnetic_Field_Bz_reflection=smooth_edge(Magnetic_Field_Bz,reflection_mask,edge_length=2*cells_per_lambda_new)[n_field_x//2:,n_field_y//2:]

Field_longitudinal_rotate_continuation,Field_transversal_rotate_continuation,Field_center_mask=rotate_parallel_field_2D(Electric_Field_Ex_reflection,Electric_Field_Ey_reflection,-np.pi/4)
Field_z_rotate_continuation=rotate_perpendicular_field_2D(Magnetic_Field_Bz_reflection,-np.pi/4)

write_field_2D(Field_list=[Field_longitudinal_rotate_continuation[Field_center_mask],Field_transversal_rotate_continuation[Field_center_mask],Field_z_rotate_continuation[Field_center_mask]],
               x_axis=x_axis_new[n_field_x//2:],y_axis=y_axis_new[round(n_field_y/4):round(n_field_y*3/4)],
               name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],nc_name=os.path.join(working_dir,'Field_%0.4d_transmission_250cpl.nc' %(i)))

exit(0)
"transversal"
"longitudinal"
"incident"
'reflection,reflected'
'transmission,transmitted'