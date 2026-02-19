from line_profiler import profile
import os
import numpy as np
import jax
from wrapt import partial
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from typing import List, Optional, Tuple, Union
import string
import xarray as xr
def outer_product(arrays: List[jnp.ndarray]) -> jnp.ndarray:
    """
    Compute the outer product of multiple 1D arrays.
    For n arrays with shapes (d0,), (d1,), ..., (d{n-1},), returns an n-dimensional
    array of shape (d0, d1, ..., d{n-1}) where:
    b[i0, i1, ..., i{n-1}] = arrays[0][i0] * arrays[1][i1] * ... * arrays[n-1][i{n-1}]
    Parameters:
    arrays : List[jnp.ndarray]
        List of 1D numpy arrays (lengths may differ)
    Returns:
    jnp.ndarray
        n-dimensional outer product array with shape (d0, d1, ..., d{n-1})
    """
    ndim = len(arrays)
    assert ndim >= 1, "At least one array is required"
    new_shape=[]
    for i, arr in enumerate(arrays):
        arrays[i] = jnp.asarray(arrays[i])
        assert arrays[i].ndim == 1, f"All input arrays must be 1D, but array {i} has shape {arrays[i].shape}"
        new_shape.append(arrays[i].size)
    new_shape=tuple(new_shape)
    #Compute outer product using numpy broadcasting.
    #More memory efficient than einsum for large arrays.
    result = arrays[0].copy()
    if ndim >1:
        for arr in arrays[1:]:
            # Add new axis to result and arr for broadcasting
            result = result[:, jnp.newaxis] * arr[jnp.newaxis, :]
            # Reshape to combine new dimension
            result = result.reshape(-1)   #shape=(d1*d2*...*d(i-1)*di, ) after multiplying with arr of shape (di,)
    return result.reshape(new_shape)
    # Use einsum for efficient computation
    # Build einsum string: for n arrays, we need n subscript letters
    letters = string.ascii_lowercase[:len(arrays)]
    einsum_str = ','.join(letters) + '->' + letters
    return jnp.einsum(einsum_str, *arrays, optimize='optimal')

@jax.jit
def sine_step(x):
    x_eff=jnp.clip(x, -0.5, 0.5)
    y = (1 + jnp.sin(jnp.pi * x_eff)) / 2
    return y
@jax.jit
def sine_step_edge(x,edge_length,edge_start,direction=1):
    """
    Sine step function for edge smoothing.
    Args:
        x (jnp.ndarray): Input array.
        edge_length (int): Length of the edge region.
        edge_start (int): Starting index of the edge region where the edge = 0.
        direction (int): 1 for rising edge, -1 for falling edge.
        When direction=1, the edge starts from 0 at edge_start and rises to 1 at edge_start + edge_length.
        When direction=-1, the edge starts from 1 at edge_start - edge_length and falls to 0 at edge_start.
    Returns:
        jnp.ndarray: Sine step values.
    """
    x_normalized=jnp.sign(direction)*(x - edge_start)/edge_length-0.5
    y=sine_step(x_normalized)
    return y

def get_edge_smooth_window(shape: Tuple[int, ...], edge_length: Union[List[int], Tuple[int, ...],jnp.ndarray]):
    """
    shape: tuple of ints, the shape of the window to be generated
    edge_length: list or tuple of ints, the length of the edge region for each dimension. 
    len(shape) must be equal to len(edge_length). If edge_length[i] = 0, no smoothing is applied along dimension i.
    scipy.signal.windows.tukey
    """
    ndim=len(shape)
    assert len(edge_length)==ndim, f"edge_length length {len(edge_length)} must match shape length {ndim}."
    window_dim_i_list: List[jnp.ndarray]=[]
    for dim_i in range(ndim):
        if edge_length[dim_i]>0 and shape[dim_i]>1:
            id_axis_i=jnp.arange(shape[dim_i])
            window_dim_i=sine_step_edge(x=id_axis_i, edge_length=edge_length[dim_i], edge_start=0, direction=1)*sine_step_edge(x=id_axis_i, edge_length=edge_length[dim_i], edge_start=shape[dim_i], direction=-1)
            window_dim_i_list.append(window_dim_i)
        else:
            window_dim_i_list.append(jnp.ones(shape[dim_i]))
    window=outer_product(window_dim_i_list)
    return window

def smooth_edge(Field:jnp.ndarray,edge_length: Union[List[int], Tuple[int, ...],int,jnp.ndarray]=10):
    shape=Field.shape
    ndim=Field.ndim
    edge_length=jnp.round(jnp.asarray(edge_length).flatten()).astype(jnp.int64)
    if edge_length.size==1:
        edge_length=jnp.full((ndim,),edge_length[0],dtype=jnp.int64)
    assert edge_length.size==ndim, f"edge_length size {edge_length.size} must match Field ndim {ndim}."
    window=get_edge_smooth_window(shape=shape, edge_length=edge_length)
    Field_smooth=Field*window
    return Field_smooth


def stack_Fields(Field_x: Optional[jnp.ndarray]=None, Field_y: Optional[jnp.ndarray]=None, Field_z: Optional[jnp.ndarray]=None):
    """
    Stack field components into a single array.

    Parameters
    ----------
    Field_x, Field_y, Field_z : jnp.ndarray
        Field components, shape (Nx, Ny, Nz)

    Returns
    -------
    Field : jnp.ndarray
        Stacked field, shape (3, Nx, Ny, Nz)
    """
    Field_input_list=[Field_x, Field_y, Field_z]
    assert any(Field_comp is not None for Field_comp in Field_input_list), "At least one field component must be provided."
    Field_i=None
    _Field_input_list=[]
    for Field_comp in Field_input_list:
        if Field_comp is not None:
            Field_i=jnp.asarray(Field_comp)
            _Field_input_list.append(Field_i)
        else:
            _Field_input_list.append(None)
    Field_input_list=_Field_input_list
    dim=Field_i.ndim
    assert dim==1 or dim==2 or dim==3, f"Field component must have at least 1 dimension, got {dim}."
    Field_stack_list=[]
    if dim==1:
        Nx=Field_i.shape[0]
        Ny=1
        Nz=1
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros((Nx,Ny,Nz)))
            else:
                assert Field_comp.shape == (Nx,), f"Field component shape {Field_comp.shape} does not match expected shape {(Nx,)}."
                Field_stack_list.append(jnp.reshape(Field_comp, (Nx,Ny,Nz)))
    elif dim==2:
        Nx=Field_i.shape[0]
        Ny=Field_i.shape[1]
        Nz=1
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros((Nx,Ny,Nz)))
            else:
                assert Field_comp.shape == (Nx,Ny), f"Field component shape {Field_comp.shape} does not match expected shape {(Nx,Ny)}."
                Field_stack_list.append(jnp.reshape(Field_comp, (Nx,Ny,Nz)))
    elif dim==3:
        for Field_comp in Field_input_list:
            if Field_comp is None:
                Field_stack_list.append(jnp.zeros_like(Field_i))
            else:
                assert Field_comp.shape == Field_i.shape, f"Field component shape {Field_comp.shape} does not match expected shape {Field_i.shape}."
                Field_stack_list.append(Field_comp)
    Field_stack=jnp.stack(Field_stack_list, axis=0)
    print(f"Stacked field shape: {Field_stack.shape}")
    return Field_stack   #shape=(3, Nx, Ny, Nz)
@jax.jit
def get_norm(Field:jnp.ndarray):
    """
    Get the norm of the field.
    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (3, Nx, Ny, Nz)

    Returns
    -------
    Field_norm : jnp.ndarray
        Norm of the field, shape (Nx, Ny, Nz)
    """
    Field_norm = jnp.linalg.norm(Field, axis=0)   #shape=(Nx, Ny, Nz)
    return Field_norm

@profile
def get_relative_divergence(
    Field:jnp.ndarray, 
    coordinate_list: List[jnp.ndarray],
    threshold=1e-3
    ):
    """
    Compute ∇·Field in real space.

    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (ndim, N_dim0, N_dim1, N_dim2), where ndim is the number of dimensions (1, 2, or 3) and N_dim0, N_dim1, N_dim2 are the sizes of each dimension. The first dimension corresponds to the field components (e.g., (Fx, Fy, Fz)).
    coordinate_list : List[jnp.ndarray]
        List of coordinate arrays for each dimension, e.g., [x_coordinate, y_coordinate, z_coordinate]. Each coordinate array should have shape (N_dim_i,). The length of coordinate_list should match the number of dimensions in Field.
    threshold : float
        The field with norm below this threshold*field_norm_max will be considered as zero to avoid numerical instability in divergence calculation.
    Returns
    -------
    divF : jnp.ndarray
        Divergence of the field, shape (Nx, Ny, Nz), unit: (units of Field)/m
    """
    Field=jnp.asarray(Field)   #shape=(ndim, N_dim0, N_dim1, N_dim2)
    ndim=Field.shape[0]
    div_F=jnp.zeros_like(Field[0])   #shape=(N_dim0, N_dim1, N_dim2), initialize divergence array
    assert ndim==len(Field.shape)-1, f"The first dimension of Field {Field.shape[0]} must match the number of dimensions {len(Field.shape)-1}."
    assert ndim==len(coordinate_list), f"Field has {ndim} dimensions but coordinate_list has {len(coordinate_list)} arrays."
    for dim_i in range(ndim):
        coordinate_i=jnp.asarray(coordinate_list[dim_i],dtype=jnp.float64).flatten()
        assert coordinate_i.size == Field.shape[dim_i+1], f"Coordinate array size {coordinate_i.size} does not match field size {Field.shape[dim_i+1]} for dimension {dim_i}."
        N_i=coordinate_i.size
        assert N_i > 0, f"Coordinate array for dimension {dim_i} must have at least one point."
        if N_i > 1:
            dFi_di = jnp.gradient(Field[dim_i], coordinate_i, axis=dim_i)   #shape=(N_dim0, N_dim1, N_dim2), ∂Fi/∂i
        else:
            dFi_di = jnp.zeros_like(Field[dim_i])
        div_F += dFi_di
    F_norm = get_norm(Field)   #shape=(Nx, Ny, Nz)
    F_norm_max=jnp.max(jnp.abs(F_norm))
    div_F_relative = jnp.where(F_norm>F_norm_max*threshold, div_F/F_norm, 0.0)   #shape=(Nx, Ny, Nz), set divergence to zero where field norm is small to avoid numerical instability
    return div_F_relative

@profile
def check_divergence(Field:jnp.ndarray, x_coordinate=[0],y_coordinate=[0],z_coordinate=[0],  threshold=5e-1,scale_length=1e-6):
    """
    Check ∇·Field = 0 in real space.
    The denser the grid, the more accurate the calculation.
    Parameters
    ----------
    Field : jnp.ndarray
        Field, shape (3, Nx, Ny, Nz)
    x_coordinate, y_coordinate, z_coordinate : list or jnp.ndarray
        Grid axes for x, y, z directions
    threshold : float
        Relative divergence tolerance
    scale_length : float, unit: m
        Characteristic length scale for normalization

    Raises
    ------
    warning if divergence exceeds threshold
    """
    x_coordinate=jnp.array(x_coordinate).flatten()
    y_coordinate=jnp.array(y_coordinate).flatten()
    z_coordinate=jnp.array(z_coordinate).flatten()
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    Nz=z_coordinate.size
    assert Nx>0 and Ny>0 and Nz>0, "Grid axes must have at least one point."
    div_F_relative=get_relative_divergence(Field=Field, coordinate_list=[x_coordinate,y_coordinate,z_coordinate], threshold=threshold)
    div_F_relative_max=jnp.max(jnp.abs(div_F_relative))
    div_err=div_F_relative_max*scale_length
    print(f"Max relative divergence error: L*|∇·F|/|F| = {div_err:.2e}")
    if div_err > threshold:
        print(f"Warning: Initial Field is not divergence-free: relative L*|∇·F|/|F| = {div_err} > {threshold}")
        return False
    else:
        print(f"Initial Field divergence check passed: relative L*|∇·F|/|F| = {div_err} <= {threshold}")
        return True

@partial(jax.jit, static_argnames=('axis'))
def square_sum(x,axis):
    return jnp.real(jnp.sum(jnp.conj(x)*x,axis=axis))
@profile
def square_integral_field(Field:jnp.ndarray,dr=[1],axis= None):
    """
    Args:
        Field (jnp.ndarray): _description_
        dr (list): [dx,dy,dz], grid spacing in each dimension. If None, defaults to ones.
        axis (int or tuple, optional): Axis or axes along which to compute the integral. If None, computes over all axes. Defaults to None.
        complex_array (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    dr=jnp.asarray(dr,dtype=jnp.float64)
    square_integral=square_sum(Field, axis=axis)*jnp.prod(dr)
    print('∬|Field|^2×dr=',square_integral,flush=True)
    return square_integral

@jax.jit
def _get_coordinate_id_float(coordinate_start,dr,N, pos):
    id_float=(pos - coordinate_start)/dr
    return id_float

@jax.jit
def _get_coordinate_id_int(coordinate_start,dr,N, pos):
    id_float=_get_coordinate_id_float(coordinate_start,dr,N, pos)
    id_int=jnp.clip(jnp.round(id_float),0,N-1).astype(jnp.int64)
    return id_int


@profile
def get_coordinate_id_float(coordinate:jnp.ndarray, pos:float|jnp.ndarray):
    """
        Get the floating-point index of a position in a given coordinate array.
        Overflow and underflow are allowed.
    Args:
        coordinate (jnp.ndarray): _description_
        pos (float | jnp.ndarray): _description_

    Returns:
        _type_: _description_
    """
    coordinate=jnp.asarray(coordinate).flatten()
    pos=jnp.asarray(pos)
    assert coordinate.ndim==1, "Coordinate must be 1D array."
    N=coordinate.size
    assert N>=1, "Coordinate must have at least one point."
    if N==1:
        return jnp.zeros_like(pos,dtype=jnp.float64)
    else:
        return _get_coordinate_id_float(coordinate_start=coordinate[0], dr=(coordinate[-1]-coordinate[0])/(N-1), N=N, pos=pos)


@profile
def get_coordinate_id_int(coordinate:jnp.ndarray, pos:float|jnp.ndarray):
    """
        Get the integer index of a position in a given coordinate array.
        Overflow and underflow are clipped to the valid range.
    Args:
        coordinate (jnp.ndarray): _description_
        pos (float | jnp.ndarray): _description_

    Returns:
        _type_: _description_
    """
    coordinate=jnp.asarray(coordinate).flatten()
    pos=jnp.asarray(pos).flatten()
    assert coordinate.ndim==1, "Coordinate must be 1D array."
    N=coordinate.size
    assert N>=1, "Coordinate must have at least one point."
    if N==1:
        id=jnp.zeros_like(pos,dtype=jnp.int64)
    else:
        id=_get_coordinate_id_int(coordinate_start=coordinate[0], dr=(coordinate[-1]-coordinate[0])/(N-1), N=N, pos=pos)
    return id

@jax.jit
def get_closest_coordinate_id(coordinate:jnp.ndarray,pos:jnp.ndarray):
    """Find the index of the closest coordinate for each position.
    Coordinate is not required to be sorted or equally spaced.
    Args:
        coordinate (jnp.ndarray): 1D array of coordinates.
        pos (jnp.ndarray): Array of positions. Can be a scalar or an array of any shape.

    Returns:
        jnp.ndarray: Array of indices of the closest coordinates.
    """
    pos=jnp.asarray(pos)
    distance=coordinate[:,*[jnp.newaxis]*pos.ndim]-pos[jnp.newaxis,...]  # shape=(len(coordinate), *pos.shape),ndim=pos.ndim+1 
    closest_coordinate_id=jnp.argmin(jnp.abs(distance), axis=0)  # shape=pos.shape,ndim=pos.ndim
    return closest_coordinate_id.astype(int)


@profile
def calculate_center(Field:jnp.ndarray, x_coordinate=[0], y_coordinate=[0], z_coordinate=[0],axis=None):
    """
    Calculate the center of the field distribution.

    Parameters
    ----------
    Field : jnp.ndarray
    x_coordinate, y_coordinate, z_coordinate : list or jnp.ndarray
        Grid axes for x, y, z directions
    axis : int or tuple, optional
        Axis or axes along which to compute the center. If None, computes over all axes. Defaults to None.

    Returns
    -------
    center : jnp.ndarray
        Center coordinates, shape depends on axis parameter
    """
    x_coordinate=jnp.array(x_coordinate).flatten()
    y_coordinate=jnp.array(y_coordinate).flatten()
    z_coordinate=jnp.array(z_coordinate).flatten()
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    Nz=z_coordinate.size
    Field=jnp.asarray(Field)
    r=jnp.meshgrid(x_coordinate, y_coordinate, z_coordinate, indexing='ij')  #shape=(3, Nx, Ny, Nz)
    weight=jnp.square(jnp.abs(Field))
    if axis is None:
        assert Field.shape == (Nx, Ny, Nz), f"Field shape {Field.shape} does not match grid shape {(Nx, Ny, Nz)}"
    else:
        axis=jnp.asarray(axis,dtype=jnp.int32)
        assert tuple(Field.shape[ax] for ax in axis) ==(Nx, Ny, Nz), f"Field shape {Field.shape} does not match grid shape {(Nx, Ny, Nz)} along specified axes"
    rc=jnp.average(a=r, weights=weight, axis=axis)
    print(f"Center of the field distribution: {rc}")
    return rc

def pad_field(
    field: jnp.ndarray,
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
    fill_value: float = 0
    ):
    # Validate inputs
    ndim = field.ndim
    assert len(output_shape) == ndim,f"output_shape length {len(output_shape)} must match array ndim {ndim}"
    # Default location is all zeros (upper-left corner)
    if location is None:
        location = tuple(0 for _ in range(ndim))
    assert len(location) == ndim,f"location length {len(location)} must match array ndim {ndim}"
    
    # Initialize output array with fill_value
    field_pad = jnp.full(output_shape, fill_value, dtype=field.dtype)
    
    # Calculate source and destination slices
    src_slices : List[slice] = []
    dst_slices : List[slice] = []
    dst_axis_start=[]
    dst_axis_end=[]
    for i in range(ndim):
        src_start = 0
        src_end = field.shape[i]
        dst_start = location[i]
        dst_end = location[i] + field.shape[i]
        dst_axis_start.append(location[i])
        dst_axis_end.append(location[i] + output_shape[i])
        # Clip to output bounds
        if dst_start < 0:
            # Negative start means part of source is clipped
            src_start = -dst_start
            dst_start = 0
        if dst_end > output_shape[i]:
            # End beyond output shape means part of source is clipped
            src_end = field.shape[i] - (dst_end - output_shape[i])
            dst_end = output_shape[i]
        # Check if there's any overlap
        if src_start >= src_end or dst_start >= dst_end:
            # No overlap along this dimension, skip copying
            src_slices.append(slice(0, 0))
            dst_slices.append(slice(0, 0))
        else:
            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))
    dst_slices=tuple(dst_slices)
    src_slices=tuple(src_slices)
    pad_slices=dst_slices
    # Copy the overlapping region
    if all(s.start < s.stop for s in src_slices):
        field_pad = field_pad.at[dst_slices].set(field[src_slices])
    print('Input shape:', field.shape)
    print('Output shape:', field_pad.shape)
    return field_pad, pad_slices

def pad_coordinates(
    coordinate_list: List[jnp.ndarray],
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
) -> List[jnp.ndarray]:
    """
    Pad coordinate arrays to match the output shape of the padded field.
    Parameters:
    -----------
    coordinate_list : List[jnp.ndarray]
        List of 1D coordinate arrays corresponding to each axis of the input array.
        Each array must be 1D with length equal to the corresponding dimension of the input field.
    output_shape : Tuple[int]
        Desired shape of the output array. Must have the same length as coordinate_list.
    location : Optional[Tuple[int]], default=None
        Starting index for each axis where the input field will be placed in the output.
        If None, defaults to (0, 0, ...) (upper-left corner).
        If location[i] < 0, it means padding after the data along coordinate[i], and the data before location[i] will not be shown in the output array.

    Returns:
    --------
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays with lengths matching output_shape
    """
    ndim = len(coordinate_list)
    assert ndim == len(output_shape), f"coordinate_list length {ndim} must match output_shape length {len(output_shape)}"
    if location is None:
        location = tuple(0 for _ in range(ndim))
    assert len(location) == ndim, f"location length {len(location)} must match coordinate_list length {ndim}"
    coordinate_pad_list = []
    for i in range(ndim):
        coordinate_i=jnp.asarray(coordinate_list[i]).flatten()
        Ni = coordinate_i.size
        dxi = (coordinate_i[-1] - coordinate_i[0]) / (Ni - 1) if Ni > 1 else 0.0
        start_id = -location[i]
        end_id = start_id + output_shape[i]
        coordinate_pad_list.append(jnp.linspace(start=coordinate_i[0] + start_id * dxi, stop=coordinate_i[0] + (end_id - 1) * dxi, num=output_shape[i], endpoint=True, dtype=jnp.float64))
    return coordinate_pad_list


def pad_field_with_coordinate(
    field: jnp.ndarray,
    coordinate_list: List[jnp.ndarray],
    output_shape: Tuple[int],
    location: Optional[Tuple[int]] = None,
    fill_value: float = 0
) -> Tuple[jnp.ndarray, List[jnp.ndarray], Tuple[slice]]:
    """
    Pad a N-dimensional array to a specified output shape at given location with a fill value.
    Parameters:
    -----------
    field : jnp.ndarray
        Input array to be padded
    coordinate_list : List[jnp.ndarray]
        List of 1D coordinate arrays corresponding to each axis of the input array.
        Each array must be 1D with length equal to the corresponding dimension of 'field'.
    output_shape : Tuple[int]
        Desired shape of the output array. Must have the same length as 'field.ndim'.
    location : Optional[Tuple[int]], default=None
        Starting index for each axis where the input array will be placed in the output.
        If None, defaults to (0, 0, ...) (upper-left corner).
        If location[i] < 0, it means padding after the data along coordinate[i], and the data before location[i] will not be shown in the output array.
    fill_value : float, default=0
        Value used for padding areas outside the input array.
    
    Returns:
    --------
    field_pad : jnp.ndarray
        Padded array with shape 'output_shape'
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays with lengths matching 'output_shape'
    pad_slices : Tuple[slice]
        Tuple of slices indicating the location of the original data within the padded array
    """
    field_pad, pad_slices = pad_field(field=field, output_shape=output_shape, location=location, fill_value=fill_value)
    coordinate_pad_list = pad_coordinates(coordinate_list=coordinate_list, output_shape=output_shape, location=location)
    return field_pad, coordinate_pad_list, pad_slices

def smooth_and_pad_fields(
    field_list: List[jnp.ndarray], 
    coordinate_list: List[jnp.ndarray],
    pad_width:tuple[int, ...],edge_length:tuple[int, ...],
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray], Tuple[slice]]:
    """
    Smooth the edges of the input fields and pad them to a specified width.

    Parameters
    ----------
    field_list : List[jnp.ndarray]
        List of input fields to be processed.
    coordinate_list : List[jnp.ndarray]
        List of grid axes for each dimension
    pad_width : tuple of ints or None
        If not None, specifies the number of points to pad on each axis (pad_x, ...).
        The padding will be applied symmetrically on both sides of each axis.
        If None, no padding is applied.
    edge_length : tuple of ints or None
        If not None, specifies the length of the edge smoothing region on each axis (edge_x, ...).
        The smoothing will be applied to the edges of the input fields before padding.
        If None, no smoothing is applied.

    Returns
    -------
    field_pad_list : List[jnp.ndarray]
        List of smoothed and padded fields
    coordinate_pad_list : List[jnp.ndarray]
        List of padded coordinate arrays corresponding to each axis
    pad_slices : Tuple[slice]
        Tuple of slices indicating the location of the original data within the padded array
    """
    shape=field_list[0].shape
    assert len(coordinate_list) == len(shape), f"coordinate_list length {len(coordinate_list)} must match field dimensions {len(shape)}"
    assert len(pad_width) == len(shape), f"pad_width length {len(pad_width)} must match field dimensions {len(shape)}"
    assert len(edge_length) == len(shape), f"edge_length length {len(edge_length)} must match field dimensions {len(shape)}"
    window=get_edge_smooth_window(shape=shape, edge_length=edge_length)
    field_pad_list=[]
    for field in field_list:
        assert field.shape == shape, f"All fields must have the same shape, but got {field.shape} and {shape}"
        field_smooth=field*window
        field_pad,pad_slices=pad_field(field=field_smooth, output_shape=tuple(s+2*p for s,p in zip(shape,pad_width)), location=pad_width, fill_value=0)
        field_pad_list.append(field_pad)
    coordinate_pad_list=pad_coordinates(coordinate_list=coordinate_list, output_shape=tuple(s+2*p for s,p in zip(shape,pad_width)), location=pad_width)
    return field_pad_list, coordinate_pad_list, pad_slices
        
