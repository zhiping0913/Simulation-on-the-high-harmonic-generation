import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import scipy.constants as C
import os
import xarray as xr
import matplotlib.pyplot as plt

N=350
a0=20
ND_a0=2
L=0.05
D=ND_a0*a0/N-L
Kappa=-0.005
working_dir=os.path.join(f'/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0={a0}/2D/Initialize_Target/K_{Kappa:+.3f},D_{D:0.2f},L_{L:0.2f}')
os.makedirs(name=working_dir,exist_ok=True)
os.chdir(path=working_dir)
print(working_dir)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2

vacuum_length_x_lambda=50   #lambda
vacuum_length_y_lambda=50   #lambda
cells_per_lambda_x =1000
cells_per_lambda_y =500

nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
ny=round(2*vacuum_length_y_lambda*cells_per_lambda_y)
print(nx,ny)

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda

dx=(x_max-x_min)/nx
dy=(y_max-y_min)/ny

#xb: Grid x at the boundary of the cell.
#x: Grid x at the center of the cell.
#Ex uses (xb,y)
#Ey uses (x,yb)
#Ez uses (x,y)
#Bx uses (x,yb)
#By uses (xb,y)
#Bz uses (xb,yb)
#Number density uses (x,y)
x_axis=jnp.linspace(start=x_min,stop=x_max,num=nx,endpoint=False)+dx/2
y_axis=jnp.linspace(start=y_min,stop=y_max,num=ny,endpoint=False)+dy/2
#xb_axis=x_axis+dx/2
#yb_axis=y_axis+dy/2
x,y=jnp.meshgrid(x_axis,y_axis,indexing='ij')
#xb,yb=jnp.meshgrid(xb_axis,yb_axis,indexing='ij')

def generate_targets(
    N_charge=350.0,
    D=1.0,
    L=0.0,
    W=1.0,
    charge=1,
    Kappa=0.0,
    ):
    """
    Args:
        N_charge: Charge density (absolute value). Unit: Nc.
        D: Target thickness in x direction. Unit: λ0.
        W:Target full width in y direction. Unit: λ0.
        L: Pre-plasma length. Unit: λ0.
        charge: Charge of one particle. Unit: e. For electron, charge=-1.
        Kappa: Curvature of the target. Unit: 1/λ0. Kappa=0 for flat target.
    """
    target_curvature=Kappa/laser_lambda   #unit: m^(-1)
    target_thickness=D*laser_lambda   #unit: m
    target_width=W*laser_lambda   #unit: m
    target_preplasma_length=L*laser_lambda   #unit: m
    target_number_density=N_charge*laser_Nc/jnp.abs(charge)   #unit: m^(-3)
    N_target=jnp.zeros(shape=(nx,ny),dtype=jnp.float64)
    parabola=x-(target_curvature/2)*jnp.square(y)
    R=1/Kappa   #Unit: λ0.
    circle=jnp.sqrt(jnp.square(x-R*laser_lambda)+jnp.square(y))-jnp.abs(R*laser_lambda)
    if L>0:
        #N_target=N_target+target_number_density*jnp.exp((parabola+target_thickness/2)/target_preplasma_length)*(parabola<-target_thickness/2)*(jnp.abs(y)<target_width/2)
        N_target=N_target+target_number_density*jnp.exp((circle+target_thickness/2)/target_preplasma_length)*(circle<-target_thickness/2)*(jnp.abs(y)<target_width/2)
    #N_target=N_target+target_number_density*(parabola>=-target_thickness
    N_target=N_target+target_number_density*(jnp.abs(circle)<(target_thickness/2))*(jnp.abs(y)<target_width/2)
    Initialize_Target=f"""
Add target.
shape: ({nx},{ny})
vacuum_length_x_lambda: {vacuum_length_x_lambda}
vacuum_length_y_lambda: {vacuum_length_y_lambda}
cells_per_lambda_x: {cells_per_lambda_x}
cells_per_lambda_y: {cells_per_lambda_y}
laser_lambda: {laser_lambda} (m)
N_charge: {N_charge} (Nc)
charge: {charge}
D: {D} (λ0)
L: {L} (λ0)
Kappa: {Kappa} (1/λ0)
"""
    print(Initialize_Target)
    with open(file=os.path.join(working_dir,'Initialize_Target.txt'),mode='a+',encoding='UTF-8') as txt:
        txt.write(Initialize_Target)
    return N_target




Ne=generate_targets(N_charge=N,D=D,W=70,L=0,charge=-1,Kappa=Kappa)
Ni=generate_targets(N_charge=N,D=D,W=70,L=0,charge=6,Kappa=Kappa)

def write_field(field:jnp.ndarray,x_axis=x_axis,y_axis=y_axis,name=''):
    np.array(field.transpose(1,0),dtype=np.float64).tofile(os.path.join(working_dir,name))
    print(os.path.join(working_dir,name))
    return 0
    field_ds=xr.Dataset(
        data_vars={
            name:(["x", "y"], field),
            },
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)))
    
    return 0


write_field(field=Ne,x_axis=x_axis,y_axis=y_axis,name='Ne')
write_field(field=Ni,x_axis=x_axis,y_axis=y_axis,name='Ni')

exit(0)

fig,ax = plt.subplots()
ax.plot(x_axis/laser_lambda,Ne[:,ny//2]/laser_Nc,label='y=0')
ax.set_xlabel('x/λ0')
ax.set_ylabel('Ne/Nc')
ax.set_title('Initial Ne')
plt.savefig(os.path.join(working_dir,'Ne_0000_y=0.png'))
print(os.path.join(working_dir,'Ne_0000_y=0.png'))
plt.close(fig)
plt.clf()
fig,ax = plt.subplots()
ax.plot(y_axis/laser_lambda,Ne[nx//2,:]/laser_Nc,label='x=0')
ax.set_xlabel('y/λ0')
ax.set_ylabel('Ne/Nc')
ax.set_title('Initial Ne')
plt.savefig(os.path.join(working_dir,'Ne_0000_x=0.png'))
print(os.path.join(working_dir,'Ne_0000_x=0.png'))
plt.close(fig)
plt.clf()

fig,ax = plt.subplots()
pcm=ax.pcolormesh(x_axis/laser_lambda,y_axis/laser_lambda,Ne.T/laser_Nc,cmap='Reds')
ax.set_aspect('equal')
ax.set_xlabel('x/λ0')
ax.set_ylabel('y/λ0')
ax.set_title('Initial Ne')
plt.colorbar(pcm).ax.set_ylabel('Ne/Nc')
plt.savefig(os.path.join(working_dir,'Ne_0000.png'))
print(os.path.join(working_dir,'Ne_0000.png'))
plt.close(fig)
plt.clf()
exit(0)