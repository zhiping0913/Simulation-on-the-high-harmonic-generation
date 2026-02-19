import sys

from line_profiler import profile
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from Spectral_Maxwell.kgrid import make_k_coordinate_from_r_coordinate
from Spectral_Maxwell.pretreat_fields import square_integral_field, get_coordinate_id_float
from Spectral_Maxwell.backend import fftn, ifftn

class Rotation:
    def __init__(self, phi=0.0, psi=0.0, theta=0.0):
        self.phi = phi
        self.psi = psi
        self.theta = theta

        self.R = self._rotation_matrix(phi, psi, theta)
        self.RT = self.R.T
        print(f'Rotation with angles (phi, psi, theta)=({phi}, {psi}, {theta}) radians initialized.', flush=True)

    @staticmethod
    def _rotation_matrix(phi, psi, theta):
        cph, sph = jnp.cos(phi), jnp.sin(phi)
        cps, sps = jnp.cos(psi), jnp.sin(psi)
        cth, sth = jnp.cos(theta), jnp.sin(theta)
        R=jnp.array([
            [cph*cps*cth - sph*sps,  cph*sps*cth + sph*cps, -cph*sth],
            [-sph*cps*cth - cph*sps, -sph*sps*cth + cph*cps,  sph*sth],
            [cps*sth,               sps*sth,                cth]
        ])   #shape: (3, 3)
        return R

    def get_dr(self, x_coordinate:jnp.ndarray, y_coordinate:jnp.ndarray, z_coordinate:jnp.ndarray):
        dx = x_coordinate[1] - x_coordinate[0] if x_coordinate.size > 1 else 1.0
        dy = y_coordinate[1] - y_coordinate[0] if y_coordinate.size > 1 else 1.0
        dz = z_coordinate[1] - z_coordinate[0] if z_coordinate.size > 1 else 1.0
        return dx, dy, dz
    @profile
    def rotate_f(self, A0:jnp.ndarray, R,x0_coordinate=[0], y0_coordinate=[0], z0_coordinate=[0], x1_coordinate=[0], y1_coordinate=[0], z1_coordinate=[0]):
        x0_coordinate=jnp.asarray(x0_coordinate,dtype=jnp.float64).flatten()
        y0_coordinate=jnp.asarray(y0_coordinate,dtype=jnp.float64).flatten()
        z0_coordinate=jnp.asarray(z0_coordinate,dtype=jnp.float64).flatten()
        x1_coordinate=jnp.asarray(x1_coordinate,dtype=jnp.float64).flatten()   #shape:(Nx1,)
        y1_coordinate=jnp.asarray(y1_coordinate,dtype=jnp.float64).flatten()   #shape:(Ny1,)
        z1_coordinate=jnp.asarray(z1_coordinate,dtype=jnp.float64).flatten()   #shape:(Nz1,)
        dx0= x0_coordinate[1]-x0_coordinate[0] if x0_coordinate.size>1 else 1.0
        dy0= y0_coordinate[1]-y0_coordinate[0] if y0_coordinate.size>1 else 1.0
        dz0= z0_coordinate[1]-z0_coordinate[0] if z0_coordinate.size>1 else 1.0
        dx1= x1_coordinate[1]-x1_coordinate[0] if x1_coordinate.size>1 else 1.0
        dy1= y1_coordinate[1]-y1_coordinate[0] if y1_coordinate.size>1 else 1.0
        dz1= z1_coordinate[1]-z1_coordinate[0] if z1_coordinate.size>1 else 1.0
        x1, y1, z1 = jnp.meshgrid(x1_coordinate, y1_coordinate, z1_coordinate, indexing="ij")
        r1 = jnp.stack([x1, y1, z1], axis=0)   #shape: (3, Nx1, Ny1, Nz1), the points in new coordinate
        r10 = jnp.einsum('ml,mijk->lijk', R, r1)   #shape: (3, Nx1, Ny1, Nz1), the points in original coordinate r0= RT @ r1
        Nx0=x0_coordinate.size
        ix=get_coordinate_id_float(coordinate=x0_coordinate, pos=r10[0])   #shape:(Nx1, Ny1, Nz1), the position (id,could be non-integer) of new x in original x
        Ny0=y0_coordinate.size
        iy=get_coordinate_id_float(coordinate=y0_coordinate, pos=r10[1])   #shape:(Nx1, Ny1, Nz1), the position (id,could be non-integer) of new y in original y
        Nz0=z0_coordinate.size
        iz=get_coordinate_id_float(coordinate=z0_coordinate, pos=r10[2])   #shape:(Nx1, Ny1, Nz1), the position (id,could be non-integer) of new z in original z
        assert A0.shape==(3,Nx0,Ny0,Nz0), f"Input field shape {A0.shape} does not match the provided grid axes sizes {(3,Nx0,Ny0,Nz0)}."
        A0x1=map_coordinates(input=A0[0], coordinates=jnp.array([ix, iy, iz]), order=1, mode='constant', cval=0.0)   #shape:(Nx1, Ny1, Nz1)
        A0y1=map_coordinates(input=A0[1], coordinates=jnp.array([ix, iy, iz]), order=1, mode='constant', cval=0.0)
        A0z1=map_coordinates(input=A0[2], coordinates=jnp.array([ix, iy, iz]), order=1, mode='constant', cval=0.0)
        A10=jnp.stack([A0x1,A0y1,A0z1],axis=0)   #shape:(3, Nx1, Ny1, Nz1)
        A1=jnp.einsum('lm,mijk->lijk', R, A10)   #shape:(3, Nx1, Ny1, Nz1)   R @ A10
        print(f'Input field shape: {A0.shape}, Output field shape: {A1.shape}',flush=True)
        I0=square_integral_field(Field=A0, dr=[dx0,dy0,dz0])
        print(f'Input field integral: {I0}',flush=True)
        I1=square_integral_field(Field=A1, dr=[dx1,dy1,dz1])
        print(f'Output field integral: {I1}',flush=True)
        print(f'Integral ratio I1/I0: {I1/I0}',flush=True)
        return A1
    @profile
    def rotate_r_space(self, A, x0_coordinate=[0], y0_coordinate=[0], z0_coordinate=[0], x1_coordinate=[0], y1_coordinate=[0], z1_coordinate=[0],direction='0->1'):
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        if direction=='1->0':
            A0=self.rotate_f(A0=A, R=self.RT, x0_coordinate=x1_coordinate, y0_coordinate=y1_coordinate, z0_coordinate=z1_coordinate, x1_coordinate=x0_coordinate, y1_coordinate=y0_coordinate, z1_coordinate=z0_coordinate)
            return A0
        else:
            A1=self.rotate_f(A0=A, R=self.R, x0_coordinate=x0_coordinate, y0_coordinate=y0_coordinate, z0_coordinate=z0_coordinate, x1_coordinate=x1_coordinate, y1_coordinate=y1_coordinate, z1_coordinate=z1_coordinate)
            return A1
    @profile
    def rotate_k_space(self, A, x_coordinate=[0], y_coordinate=[0], z_coordinate=[0],direction='0->1'):
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        kx_coordinate, dkx, dx = make_k_coordinate_from_r_coordinate(x_coordinate)
        ky_coordinate, dky, dy = make_k_coordinate_from_r_coordinate(y_coordinate)
        kz_coordinate, dkz, dz = make_k_coordinate_from_r_coordinate(z_coordinate)
        Ak = fftn(A, axes=(1,2,3))   #shape: (3, Nx, Ny, Nz)
        if direction=='1->0':
            A0k=self.rotate_f(A0=Ak, R=self.RT, x0_coordinate=kx_coordinate, y0_coordinate=ky_coordinate, z0_coordinate=kz_coordinate, x1_coordinate=kx_coordinate, y1_coordinate=ky_coordinate, z1_coordinate=kz_coordinate)
            return ifftn(A0k, axes=(1,2,3)).real
        else:
            A1k=self.rotate_f(A0=Ak, R=self.R, x0_coordinate=kx_coordinate, y0_coordinate=ky_coordinate, z0_coordinate=kz_coordinate, x1_coordinate=kx_coordinate, y1_coordinate=ky_coordinate, z1_coordinate=kz_coordinate)
            return ifftn(A1k, axes=(1,2,3)).real
    @profile
    def rotate(self, A, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate=[0], y1_coordinate=[0], z1_coordinate=[0],direction='0->1',space='r'):
        """
        Rotate field using selected space.
        space="r":
            Rotate in real space,
            requires (x0,y0,z0) and (x1,y1,z1)
        space="k":
            Rotate in k space,
            requires (x0,y0,z0) only
        """
        assert direction in ['0->1','1->0'], "direction must be '0->1' or '1->0'"
        assert space in ['r','k'], "space must be 'r' or 'k'"
        if space == "r":
             return self.rotate_r_space(A, x0_coordinate, y0_coordinate, z0_coordinate, x1_coordinate, y1_coordinate, z1_coordinate,direction=direction)
        elif space == "k":
            return self.rotate_k_space(A, x0_coordinate, y0_coordinate, z0_coordinate,direction=direction)


