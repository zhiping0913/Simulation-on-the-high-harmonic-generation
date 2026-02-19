import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start/Spectral_Maxwell')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from Spectral_Maxwell.backend import fftfreq
def make_k_coordinate_from_r_coordinate(r_coordinate):
    """
    Make k axis from real-space grid parameters
    dx*dk=2π/N
    """
    r_coordinate = jnp.asarray(r_coordinate,dtype=jnp.float64).flatten()
    N = r_coordinate.size
    assert N >0, "Grid axis must have at least one point."
    if N > 1:
        dr = (r_coordinate[-1] - r_coordinate[0]) / (N - 1)
        k_coordinate = fftfreq(N, d=dr) * 2 * jnp.pi
        dk = k_coordinate[1] - k_coordinate[0]
    else:
        k_coordinate = jnp.array([0.0]).flatten()
        dk = 2*jnp.pi
        dr = 1.0
    return k_coordinate, dk, dr
def make_r_coordinate_from_k_coordinate(k_coordinate):
    """
    Make real-space axis from k-space grid parameters
    dx*dk=2π/N
    """
    k_coordinate = jnp.asarray(k_coordinate,dtype=jnp.float64).flatten()
    N = k_coordinate.size
    assert N >0, "Grid axis must have at least one point."
    if N > 1:
        dk = (k_coordinate[-1] - k_coordinate[0]) / (N - 1)
        dr = 2 * jnp.pi / (N * dk)
        r_coordinate = jnp.linspace(-dr*(N-1)/2, dr*(N-1)/2, N,endpoint=True,dtype=jnp.float64)
    else:
        r_coordinate = jnp.array([0.0]).flatten()
        dr = 1.0
        dk = 2*jnp.pi
    return r_coordinate, dr, dk
class grid_k:
    """
    Make k-space grid from given real-space grid axes.:
        k      : (3, Nx, Ny, Nz)   k=(kx, ky, kz)
        k_norm : (Nx, Ny, Nz)   |k|=sqrt(kx^2+ky^2+kz^2)
        k_hat  : (3, Nx, Ny, Nz)   k_hat=k/|k|, with k_hat=0 when |k|=0
    """
    def __init__(self, x_coordinate=[0], y_coordinate=[0], z_coordinate=[0]):
        x_coordinate=jnp.asarray(x_coordinate,dtype=jnp.float64).flatten()
        y_coordinate=jnp.asarray(y_coordinate,dtype=jnp.float64).flatten()
        z_coordinate=jnp.asarray(z_coordinate,dtype=jnp.float64).flatten()
        self.Nx=x_coordinate.size
        self.Ny=y_coordinate.size
        self.Nz=z_coordinate.size
        assert self.Nx>1 or self.Ny>1 or self.Nz>1, "At least one grid axis must have more than two points."
        self.shape=(self.Nx,self.Ny,self.Nz)
        self.kx_coordinate, self.dkx, self.dx = make_k_coordinate_from_r_coordinate(x_coordinate)
        self.ky_coordinate, self.dky, self.dy = make_k_coordinate_from_r_coordinate(y_coordinate)
        self.kz_coordinate, self.dkz, self.dz = make_k_coordinate_from_r_coordinate(z_coordinate)
        self.dk=min(filter(lambda v: v>0.0, [self.dkx,self.dky,self.dkz]))
        self.k = jnp.asarray(jnp.meshgrid(self.kx_coordinate, self.ky_coordinate, self.kz_coordinate, indexing="ij"),dtype=jnp.float64)   #shape=(3, Nx, Ny, Nz), k=(kx, ky, kz)
        self.k_norm = jnp.linalg.norm(self.k,axis=0)   #shape=(Nx, Ny, Nz), |k|=sqrt(kx^2+ky^2+kz^2)
        self.k_mask = self.k_norm > self.dk/100   #shape=(Nx, Ny, Nz), avoid division by zero
        self.k_hat = jnp.where(self.k_mask[jnp.newaxis,:,:,:], self.k / self.k_norm[jnp.newaxis, :, :, :],0.0)   #shape=(3, Nx, Ny, Nz), k_hat=k/|k| with k_hat=0 when |k|=0

if __name__ == "__main__":
    x_coordinate=[-1.0, -0.5, 0.0, 0.5, 1.0]
    y_coordinate=[-1.0, -0.5, 0.0, 0.5, 1.0]
    z_coordinate=[0]
    print("Test kgrid:")
    kgrid=grid_k(x_coordinate,y_coordinate,z_coordinate)
    print("kx_coordinate:",kgrid.kx_coordinate)
    print("ky_coordinate:",kgrid.ky_coordinate)
    print("kz_coordinate:",kgrid.kz_coordinate)
