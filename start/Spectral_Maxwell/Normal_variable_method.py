from functools import partial
import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start/Spectral_Maxwell')
from typing import Optional, Union, Tuple
from line_profiler import profile
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from jax import jit, vmap
import jax.numpy as jnp
from Spectral_Maxwell.backend import fftn, ifftn
from Spectral_Maxwell.pretreat_fields import check_divergence,stack_Fields,smooth_and_pad_fields
from Spectral_Maxwell.kgrid import grid_k
import scipy.constants as C
@profile
@partial(jit)
def evolution_t(omega_dot_t,Ek0, Bk0,k_cross_Ek0,k_cross_Bk0, k_hat, window_shift_velocity=jnp.array((0.0,0.0,0.0))):
    """_summary_

    Args:
        Ek0: Initial E field in k-space, shape: (3, Nx, Ny, Nz)
        Bk0: Initial B field in k-space, shape: (3, Nx, Ny, Nz)
        omega_dot_t: ω·t, shape: (Nx, Ny, Nz). Unit: rad
        k_cross_Ek0: k_hat × Ek0, shape: (3, Nx, Ny, Nz)
        k_cross_Bk0: k_hat × Bk0, shape: (3, Nx, Ny, Nz)
        k_hat: unit k-vector grid, shape: (3, Nx, Ny, Nz)
        omega_dot_t: ω·t, shape: (Nx, Ny, Nz). Unit: rad
        window_shift_velocity: The (3d) velocity of the window following the evolution of the field. (vx,vy,vz), shape: (3,). Unit: m/s
    Returns:
        Ek_evolution_in_window: E field in k-space after evolution in moving window, shape: (3, Nx, Ny, Nz)
        Bk_evolution_in_window: B field in k-space after evolution in moving window, shape: (3, Nx, Ny, Nz)
    """
    coswt = jnp.cos(omega_dot_t)   #shape=(Nx, Ny, Nz)
    sinwt = jnp.sin(omega_dot_t)   #shape=(Nx, Ny, Nz)

    Ek_evolution = Ek0 * coswt[jnp.newaxis, :,:,:]+ 1j * k_cross_Bk0 * sinwt[jnp.newaxis, :,:,:]   #shape=(3, Nx, Ny, Nz)
    Bk_evolution = Bk0 * coswt[jnp.newaxis, :,:,:]- 1j * k_cross_Ek0 * sinwt[jnp.newaxis, :,:,:]   #shape=(3, Nx, Ny, Nz)

    window_phase_shift=jnp.einsum('lijk,l->ijk',k_hat,window_shift_velocity)*omega_dot_t/C.speed_of_light   #k_hat·v·ω·t/c=(kx,ky,kz)·(vx,vy,vz)·t, shape=(Nx, Ny, Nz). Unit: rad
    Ek_evolution_in_window=Ek_evolution * jnp.exp(1j * window_phase_shift)[jnp.newaxis, :,:,:]   #shape=(3, Nx, Ny, Nz)
    Bk_evolution_in_window=Bk_evolution * jnp.exp(1j * window_phase_shift)[jnp.newaxis, :,:,:]   #shape=(3, Nx, Ny, Nz)
    return Ek_evolution_in_window, Bk_evolution_in_window

@partial(jit)
def transverse_projection(Fk, k_hat, k_mask):
    """
    Project Fk onto divergence-free (transverse) subspace.
    Fk shape: (3, Nx, Ny, Nz)
    """
    Fk_proj = jnp.array(Fk, copy=True)   #shape=(3, Nx, Ny, Nz)
    k_dot_Fk = jnp.einsum("lijk,lijk->ijk", k_hat, Fk)   #shape=(Nx, Ny, Nz)   k_hat · Fk
    Fk_proj = jnp.where(k_mask[jnp.newaxis, :, :, :], 
                        Fk_proj- k_hat* k_dot_Fk[jnp.newaxis, :, :, :],   #shape=(3, Nx, Ny, Nz)
                        Fk_proj)   #k=0 mode
    return Fk_proj

class Spectral_Maxwell_Solver:
    """
    Exact spectral Maxwell solver (vacuum).
    Fields are assumed transverse.
    This is explicitly checked at initialization.
    """
    def __init__(
        self, 
        E0: jnp.ndarray, B0: jnp.ndarray, 
        x_coordinate=[0],y_coordinate=[0],z_coordinate=[0],
        pad_width:Optional[Union[jnp.ndarray, Tuple[int, int, int]]]=None,
        smooth_length:Optional[Union[jnp.ndarray, Tuple[int, int, int]]]=None,
        ):
        """
        Parameters
        ----------
        E0: jnp.ndarray
            Initial E fields, unit: V/m, shape (3, Nx, Ny, Nz)
        B0 : jnp.ndarray
            Initial B fields, unit: T, shape (3, Nx, Ny, Nz)
        x_coordinate : list
            x-axis grid points, unit: m, shape (Nx,)
        y_coordinate : list
            y-axis grid points, unit: m, shape (Ny,)
        z_coordinate : list
            z-axis grid points, unit: m, shape (Nz,)
        pad_width: tuple or None
            If not None, pad the input fields to the given width before initializing the solver. The pad_width should be a tuple of three integers (pad_x, pad_y, pad_z) specifying the number of points to pad on each axis. The padding will be applied symmetrically on both sides of each axis.
            If None, pad_width=(Nx,Ny,Nz)
        smooth_length: tuple or None
            If not None, apply edge smoothing to the input fields with the given length before initializing the solver. The smooth_length should be a tuple of three integers (smooth_x, smooth_y, smooth_z) specifying the length of the smoothing region on each axis.
            If None, smooth_length=(Nx//20,Ny//20,Nz//20)
        """
        self.x_coordinate = jnp.array(x_coordinate).flatten()
        self.y_coordinate = jnp.array(y_coordinate).flatten()
        self.z_coordinate = jnp.array(z_coordinate).flatten()
        self.Nx = self.x_coordinate.size
        self.Ny = self.y_coordinate.size
        self.Nz = self.z_coordinate.size
        if smooth_length is None:
            smooth_length=(self.Nx//20,self.Ny//20,self.Nz//20)
        if pad_width is None:
            pad_width=(self.Nx,self.Ny,self.Nz)
        smooth_length=jnp.asarray(smooth_length).flatten().astype(jnp.int32)
        pad_width=jnp.asarray(pad_width).flatten().astype(jnp.int32)
        assert self.Nx>0 and self.Ny>0 and self.Nz>0, "Grid axes must have at least one point."
        assert len(smooth_length)==3, "smooth_length must be a tuple of three integers."
        assert len(pad_width)==3, "pad_width must be a tuple of three integers."
        self.shape = (self.Nx, self.Ny, self.Nz)
        self.E0 = jnp.asarray(E0,copy=True)   #shape=(3, Nx, Ny, Nz)
        self.B0 = jnp.asarray(B0,copy=True)   #shape=(3, Nx, Ny, Nz)
        assert self.E0.shape == (3, self.Nx, self.Ny, self.Nz), f"E0 shape {self.E0.shape} does not match grid shape {(3, self.Nx, self.Ny, self.Nz)}"
        assert self.B0.shape == (3, self.Nx, self.Ny, self.Nz), f"B0 shape {self.B0.shape} does not match grid shape {(3, self.Nx, self.Ny, self.Nz)}"
        print(f"Initial field shapes {self.E0.shape} verified.", flush=True)
        # Check transversality in real space
        check_divergence(Field=self.E0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        check_divergence(Field=self.B0, x_coordinate=self.x_coordinate, y_coordinate=self.y_coordinate, z_coordinate=self.z_coordinate)
        self.Nx_pad, self.Ny_pad, self.Nz_pad = [self.shape[i]+2*pad_width[i] for i in range(3)]
        field_pad_list, coordinate_pad_list, pad_slices=smooth_and_pad_fields(
            field_list=[self.E0,self.B0], 
            coordinate_list=[[0,1,2],self.x_coordinate, self.y_coordinate, self.z_coordinate],
            edge_length=(0,*smooth_length), pad_width=(0,*pad_width),
        )
        self.E0_pad, self.B0_pad = field_pad_list
        _,self.x_coordinate_pad, self.y_coordinate_pad, self.z_coordinate_pad = coordinate_pad_list
        self.pad_slices = pad_slices
        print(f"Padded field shapes {self.E0_pad.shape} verified.", flush=True)
        print(f"Padded coordinate shapes {self.x_coordinate_pad.shape}, {self.y_coordinate_pad.shape}, {self.z_coordinate_pad.shape} verified.", flush=True)

        # k-space grid
        self.grid_k = grid_k(
            x_coordinate=self.x_coordinate_pad,
            y_coordinate=self.y_coordinate_pad,
            z_coordinate=self.z_coordinate_pad,
        )
        self.k = self.grid_k.k   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.k_norm = self.grid_k.k_norm   #shape=(Nx_pad, Ny_pad, Nz_pad)
        self.k_hat = self.grid_k.k_hat   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.omega = C.speed_of_light * self.k_norm   #shape=(Nx_pad, Ny_pad, Nz_pad)
        
        # FFT initial fields
        self.Ek0 = fftn(self.E0_pad, axes=(1,2,3))   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.Bk0 = fftn(self.B0_pad*C.speed_of_light, axes=(1,2,3))   #shape=(3, Nx_pad, Ny_pad, Nz_pad)

        # Enforce transversality in k-space (numerical safety)
        self.Ek0 = transverse_projection(self.Ek0, self.k_hat, self.grid_k.k_mask)
        self.Bk0 = transverse_projection(self.Bk0, self.k_hat, self.grid_k.k_mask)
        
        self.k_cross_Bk0 = jnp.cross(self.k_hat, self.Bk0, axis=0)   #shape=(3, Nx_pad, Ny_pad, Nz_pad)
        self.k_cross_Ek0 = jnp.cross(self.k_hat, self.Ek0, axis=0)   #shape=(3, Nx_pad, Ny_pad, Nz_pad)

    @profile
    def evolution(self, evolution_time=0.0,window_shift_velocity=jnp.array((0.0,0.0,0.0))):
        """_summary_

        Args:
            evolution_time: evolution time. Unit: s
            window_shift_velocity: The (3d) velocity of the window following the evolution of the field. (vx,vy,vz). Unit: m/s
        Returns:
            EB_evolution_dict={
                "E": E_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: V/m
                "B": B_evolution_in_window,   #shape=(3, Nx, Ny, Nz), unit: T
                "x_coordinate": window_x_coordinate,   #shape=(Nx,), unit: m
                "y_coordinate": window_y_coordinate,   #shape=(Ny,), unit: m
                "z_coordinate": window_z_coordinate,   #shape=(Nz,), unit: m
            }
        """
        window_shift_velocity=jnp.array(window_shift_velocity,dtype=jnp.float64)
        assert window_shift_velocity.shape==(3,)
        print(f"Evolution time: {evolution_time} s, window shift velocity: {window_shift_velocity} m/s", flush=True)
        Ek_evolution_in_window,Bk_evolution_in_window=evolution_t(
            omega_dot_t=self.omega*evolution_time,
            Ek0=self.Ek0, Bk0=self.Bk0,k_cross_Ek0=self.k_cross_Ek0,k_cross_Bk0=self.k_cross_Bk0, 
            k_hat=self.k_hat, window_shift_velocity=window_shift_velocity
            )
        window_x_coordinate=self.x_coordinate+window_shift_velocity[0]*evolution_time
        window_y_coordinate=self.y_coordinate+window_shift_velocity[1]*evolution_time
        window_z_coordinate=self.z_coordinate+window_shift_velocity[2]*evolution_time
        E_evolution_in_window = jnp.real(ifftn(Ek_evolution_in_window, axes=(1,2,3)))   #shape=(3, Nx_pad, Ny_pad, Nz_pad), unit: V/m
        B_evolution_in_window = jnp.real(ifftn(Bk_evolution_in_window, axes=(1,2,3)))/C.speed_of_light   #shape=(3, Nx_pad, Ny_pad, Nz_pad), unit: T

        return {
            "E": E_evolution_in_window[self.pad_slices],   #shape=(3, Nx, Ny, Nz), unit: V/m
            "B": B_evolution_in_window[self.pad_slices],   #shape=(3, Nx, Ny, Nz), unit: T
            "x_coordinate": window_x_coordinate,   #shape=(Nx,), unit: m
            "y_coordinate": window_y_coordinate,   #shape=(Ny,), unit: m
            "z_coordinate": window_z_coordinate,   #shape=(Nz,), unit: m
        }

class Spectral_Maxwell_Solver_1D():
    """
    1D version of Spectral_Maxwell_Solver, with fields and grid only along x axis.
    """
    def __init__(self, 
                 E0x:Optional[jnp.ndarray]=None, E0y:Optional[jnp.ndarray]=None, E0z:Optional[jnp.ndarray]=None,
                 B0x:Optional[jnp.ndarray]=None, B0y:Optional[jnp.ndarray]=None, B0z:Optional[jnp.ndarray]=None,
                 x_coordinate=[0],
                 pad_width:Optional[Union[jnp.ndarray, Tuple[int], int]] = None, 
                 smooth_length:Optional[Union[jnp.ndarray, Tuple[int], int]] = None,
                 ):
        self.x_coordinate = jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        self.Nx=self.x_coordinate.size
        E0=stack_Fields(Field_x=E0x, Field_y=E0y, Field_z=E0z)  #shape=(3, Nx,1,1)
        B0=stack_Fields(Field_x=B0x, Field_y=B0y, Field_z=B0z)  #shape=(3, Nx,1,1)
        assert E0.shape == (3, self.Nx, 1, 1), f"E0 shape {E0.shape} does not match grid shape {(3, self.Nx, 1, 1)}"
        assert B0.shape == (3, self.Nx, 1, 1), f"B0 shape {B0.shape} does not match grid shape {(3, self.Nx, 1, 1)}"
        print(f"Initial field shapes {E0.shape} verified.", flush=True)
        pad_width=jnp.asarray(pad_width).flatten()
        smooth_length=jnp.asarray(smooth_length).flatten()
        self.Solver=Spectral_Maxwell_Solver(
            E0=E0, B0=B0, 
            x_coordinate=x_coordinate, y_coordinate=[0], z_coordinate=[0],
            pad_width=jnp.pad(jnp.asarray(pad_width).flatten(),(0,2)).astype(jnp.int32), 
            smooth_length=jnp.pad(jnp.asarray(smooth_length).flatten(),(0,2)).astype(jnp.int32),
            )
    def evolution(self, evolution_time=0.0,window_shift_velocity=0.0):
        """_summary_

        Args:
            evolution_time (float, optional): _description_. Defaults to 0.0.
            window_shift_velocity: vx. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        window_shift_velocity=jnp.pad(jnp.array(window_shift_velocity,dtype=jnp.float64).flatten(), pad_width=((0, 2),))   #shape=(3,)
        EB_evolution_dict=self.Solver.evolution(
            evolution_time=evolution_time,window_shift_velocity=window_shift_velocity
            )
        E_evolution_in_window=EB_evolution_dict["E"][:, :, 0, 0]   #shape=(3, Nx)
        B_evolution_in_window=EB_evolution_dict["B"][:, :, 0, 0]   #shape=(3, Nx)
        window_x_coordinate=EB_evolution_dict["x_coordinate"]   #shape=(Nx,)
        return {
            'Ex': E_evolution_in_window[0,:],
            'Ey': E_evolution_in_window[1,:],
            'Ez': E_evolution_in_window[2,:],
            'Bx': B_evolution_in_window[0,:],
            'By': B_evolution_in_window[1,:],
            'Bz': B_evolution_in_window[2,:],
            'x_coordinate': window_x_coordinate,
        }

class Spectral_Maxwell_Solver_2D():
    """
    2D version of Spectral_Maxwell_Solver, with fields and grid only along x and y axes.
    """
    def __init__(self, 
                 E0x:Optional[jnp.ndarray]=None, E0y:Optional[jnp.ndarray]=None, E0z:Optional[jnp.ndarray]=None,
                 B0x:Optional[jnp.ndarray]=None, B0y:Optional[jnp.ndarray]=None, B0z:Optional[jnp.ndarray]=None,    
                 x_coordinate=[0], y_coordinate=[0],
                 pad_width:Optional[Union[jnp.ndarray, Tuple[int], int]] = None, 
                 smooth_length:Optional[Union[jnp.ndarray, Tuple[int], int]] = None,
                 ):
        self.x_coordinate = jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        self.y_coordinate = jnp.asarray(y_coordinate,dtype=jnp.float64).flatten()
        self.Nx=self.x_coordinate.size
        self.Ny=self.y_coordinate.size
        assert self.Nx>0 and self.Ny>0, "Grid axes must have at least one point."
        if isinstance(smooth_length, int):
            smooth_length=(smooth_length,smooth_length,0)
        elif isinstance(smooth_length, (list, tuple, jnp.ndarray)):
            smooth_length=jnp.pad(jnp.asarray(smooth_length).flatten(), pad_width=((0, 1),)).astype(jnp.int32)
        else:
            smooth_length=(self.Nx//20,self.Ny//20,0)
        assert len(smooth_length)==3
        if isinstance(pad_width, int):
            pad_width=(pad_width,pad_width,0)
        elif isinstance(pad_width, (list, tuple, jnp.ndarray)):
            pad_width=jnp.pad(jnp.asarray(pad_width).flatten(), pad_width=((0, 1),)).astype(jnp.int32)
        else:
            pad_width=(self.Nx,self.Ny,0)
        assert len(pad_width)==3
        E0=stack_Fields(Field_x=E0x, Field_y=E0y, Field_z=E0z)  #shape=(3, Nx, Ny,1)
        B0=stack_Fields(Field_x=B0x, Field_y=B0y, Field_z=B0z)  #shape=(3, Nx, Ny,1)
        assert E0.shape == (3, self.Nx, self.Ny, 1), f"E0 shape {E0.shape} does not match grid shape {(3, self.Nx, self.Ny, 1)}"
        assert B0.shape == (3, self.Nx, self.Ny, 1), f"B0 shape {B0.shape} does not match grid shape {(3, self.Nx, self.Ny, 1)}"
        print(f"Initial field shapes {E0.shape} verified.", flush=True)
        self.Solver=Spectral_Maxwell_Solver(
            E0=E0, B0=B0, 
            x_coordinate=x_coordinate, y_coordinate=y_coordinate, z_coordinate=[0],
            pad_width=pad_width, 
            smooth_length=smooth_length,
            )
    def evolution(self, evolution_time=0.0,window_shift_velocity=(0.0,0.0)):
        """_summary_

        Args:
            evolution_time (float, optional): _description_. Defaults to 0.0.
            window_shift_velocity: (vx,vy). Defaults to (0.0,0.0).

        Returns:
            _type_: _description_
        """
        window_shift_velocity=jnp.pad(jnp.array(window_shift_velocity,dtype=jnp.float64).flatten(), pad_width=((0, 1),))   #shape=(3,)
        EB_evolution_dict=self.Solver.evolution(
            evolution_time=evolution_time,window_shift_velocity=window_shift_velocity
            )
        E_evolution_in_window=EB_evolution_dict["E"][:, :, :, 0]   #shape=(3, Nx, Ny)
        B_evolution_in_window=EB_evolution_dict["B"][:, :, :, 0]   #shape=(3, Nx, Ny)
        window_x_coordinate=EB_evolution_dict["x_coordinate"]   #shape=(Nx,)
        window_y_coordinate=EB_evolution_dict["y_coordinate"]   #shape=(Ny,)
        return {
            'Ex': E_evolution_in_window[0,:,:],
            'Ey': E_evolution_in_window[1,:,:],
            'Ez': E_evolution_in_window[2,:,:],
            'Bx': B_evolution_in_window[0,:,:],
            'By': B_evolution_in_window[1,:,:],
            'Bz': B_evolution_in_window[2,:,:],
            'x_coordinate': window_x_coordinate,
            'y_coordinate': window_y_coordinate,
        }