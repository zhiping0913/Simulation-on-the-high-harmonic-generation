import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import numpy as np
import scipy.constants as C
import os
import math
import xarray as xr
import matplotlib.pyplot as plt
from start import read_sdf,read_nc,read_dat,write_field_2D
laser_lambda = 0.8*C.micron		# Laser wavelength, microns

vacuum_length_x_lambda=20   #lambda
vacuum_length_y_lambda=10   #lambda
cells_per_lambda_x =200
cells_per_lambda_y =200

nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
ny=round(2*vacuum_length_y_lambda*cells_per_lambda_y)

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
x_axis=np.linspace(start=x_min,stop=x_max,num=nx,endpoint=False)+dx/2
y_axis=np.linspace(start=y_min,stop=y_max,num=ny,endpoint=False)+dy/2
xb_axis=x_axis+dx/2
yb_axis=y_axis+dy/2
x,y=np.meshgrid(x_axis,y_axis,indexing='ij')
xb,yb=np.meshgrid(xb_axis,yb_axis,indexing='ij')

Ex=np.zeros(shape=(nx,ny),dtype=np.float64)
Ey=np.zeros(shape=(nx,ny),dtype=np.float64)
Ez=np.zeros(shape=(nx,ny),dtype=np.float64)
Bx=np.zeros(shape=(nx,ny),dtype=np.float64)
By=np.zeros(shape=(nx,ny),dtype=np.float64)
Bz=np.zeros(shape=(nx,ny),dtype=np.float64)

def generate_fields(
    laser_a0=10.0,
    laser_FWHM=25*C.femto,
    laser_w0=3*C.micro,
    laser_phase =0.0,
    theta_degree=0.0,
    center_position=0.0,
    polarisation_angle_degree=0.0,

    ):
    """
    Args:
        laser_a0: Normalized laser field strength. 
        laser_FWHM: Intensity FWHM. Unit: s.
        laser_w0: Waist radius. Unit: m.
        laser_phase: Phase at the peak of the envelope measured at the focus. cosφ. Unit: rad.
        theta_degree: Incident angle. Unit: °.
        center_position: The distance between the center of the pulse and the origin (0,0,0). Unit: m
        polarisation_angle_degree: The polarisation angle. Suppose the wave is in +x direction, polarisation angle 0° means E is in +y direction, and polarisation angle 90° means E is in +z direction. Unit: °.
    """
    global Ex,Ey,Ez,Bx,By,Bz
    theta_rad=np.radians(theta_degree)
    polarisation_angle_rad=np.radians(polarisation_angle_degree)
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2)) 
    laser_k=2*np.pi/laser_lambda   #unit: 1/m
    laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
    laser_zR=np.pi*laser_w0**2/laser_lambda  #unit: m
    def transversal_field(xx:np.ndarray,yy:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*np.square(y_rot)/2)
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=np.power(1+np.square(x_rot/laser_zR),-1/4)*np.exp(-np.square(y_rot/w_Z))*np.exp(-np.square(phi/(laser_omega*laser_tau)))*np.cos(phi-0.5*Gouy_Z+laser_phase)
        return field
    def longitudinal_E_field(xx:np.ndarray,yy:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*np.square(y_rot)/2)
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=np.power(1+np.square(x_rot/laser_zR),-3/4)*(y_rot*np.cos(polarisation_angle_rad)/laser_zR)*np.exp(-np.square(y_rot/w_Z))*np.exp(-np.square(phi/(laser_omega*laser_tau)))*np.cos(phi-1.5*Gouy_Z-np.pi/2+laser_phase)
        return field
    def longitudinal_B_field(xx:np.ndarray,yy:np.ndarray):
        x_rot=xx*np.cos(theta_rad)+yy*np.sin(theta_rad)
        y_rot=-xx*np.sin(theta_rad)+yy*np.cos(theta_rad)
        w_Z=laser_w0*np.sqrt(1+np.square(x_rot/laser_zR))
        Kappa_Z=x_rot/(np.square(x_rot)+np.square(laser_zR))   #κ=1/R is the curvature of the wavefront. unit: 1/m
        phi=laser_k*(x_rot-center_position+Kappa_Z*np.square(y_rot)/2)
        Gouy_Z=np.arctan(x_rot/laser_zR)
        field=np.power(1+np.square(x_rot/laser_zR),-3/4)*(-y_rot*np.sin(polarisation_angle_rad)/laser_zR)*np.exp(-np.square(y_rot/w_Z))*np.exp(-np.square(phi/(laser_omega*laser_tau)))*np.cos(phi-1.5*Gouy_Z-np.pi/2+laser_phase)
        return field
    Ex=Ex+laser_amp*(longitudinal_E_field(xb,y)*np.cos(theta_rad)-transversal_field(xb,y)*np.cos(polarisation_angle_rad)*np.sin(theta_rad))*(x<0)
    Ey=Ey+laser_amp*(longitudinal_E_field(x,yb)*np.sin(theta_rad)+transversal_field(x,yb)*np.cos(polarisation_angle_rad)*np.cos(theta_rad))*(x<0)
    Ez=Ez+laser_amp*transversal_field(x,y)*np.sin(polarisation_angle_rad)*(x<0)
    Bx=Bx+(laser_amp/C.speed_of_light)*(longitudinal_B_field(x,yb)*np.cos(theta_rad)+transversal_field(x,yb)*np.sin(polarisation_angle_rad)*np.sin(theta_rad))*(x<0)
    By=By+(laser_amp/C.speed_of_light)*(longitudinal_B_field(xb,y)*np.sin(theta_rad)-transversal_field(xb,y)*np.sin(polarisation_angle_rad)*np.cos(theta_rad))*(x<0)
    Bz=Bz+(laser_amp/C.speed_of_light)*transversal_field(xb,yb)*np.cos(polarisation_angle_rad)*(x<0)
    print(np.max(np.abs(np.gradient(Ex,xb_axis,axis=0)*dx+np.gradient(Ey,yb_axis,axis=1)*dy))/laser_amp)   #∂Ex/∂x+∂Ey/∂y
    print(np.max(np.abs(np.gradient(Bx,x_axis,axis=0)*dx+np.gradient(By,y_axis,axis=1)*dy))/(laser_amp/C.speed_of_light))   #∂Bx/∂x+∂By/∂y
    print(np.max(np.abs(np.gradient(By,xb_axis,axis=0)*dx-np.gradient(Bx,yb_axis,axis=1)*dy))/(laser_amp/C.speed_of_light))
    with open(file=os.path.join(working_dir,'Initialize_Field.txt'),mode='a+',encoding='UTF-8') as txt:
        txt.write(
f"""
Add field.
vacuum_length_x_lambda: {vacuum_length_x_lambda}
vacuum_length_y_lambda: {vacuum_length_y_lambda}
cells_per_lambda_x: {cells_per_lambda_x}
cells_per_lambda_y: {cells_per_lambda_y}
laser_lambda: {laser_lambda} (m)
laser_a0: {laser_a0}
laser_FWHM: {laser_FWHM} (s)
laser_w0: {laser_w0} (m)
laser_phase: {laser_phase} (rad)
theta_degree: {theta_degree} (°)
polarisation_angle_degree: {polarisation_angle_degree} (°)
"""
        )
    return Ex,Ey,Ez,Bx,By,Bz

def write_field(field:np.ndarray,x_axis=x_axis,y_axis=y_axis,name=''):
    field.transpose(1,0).tofile(os.path.join(working_dir,name))
    print(name)
    return 0
    field_ds=xr.Dataset(
        data_vars={
            name:(["x", "y"], field),
            },
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis)}
        )
    field_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)))
    
    return 0



working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_02'


generate_fields(
    laser_a0=3,
    laser_FWHM=8*C.femto,
    laser_w0=3*laser_lambda,
    theta_degree=0,
    laser_phase=np.pi/2,
    center_position=-10*laser_lambda,
)
#write_field_2D(Field_list=[Ex],x_axis=xb_axis,y_axis=y_axis,name_list=['Ex'],nc_name=os.path.join(working_dir,'Ex.nc'))
write_field(field=Ex,x_axis=xb_axis,y_axis=y_axis,name='Ex')
write_field(field=Ey,x_axis=x_axis,y_axis=yb_axis,name='Ey')
write_field(field=Ez,x_axis=x_axis,y_axis=y_axis,name='Ez')
write_field(field=Bx,x_axis=x_axis,y_axis=yb_axis,name='Bx')
write_field(field=By,x_axis=xb_axis,y_axis=y_axis,name='By')
write_field(field=Bz,x_axis=xb_axis,y_axis=yb_axis,name='Bz')

fig,ax = plt.subplots()
ax.plot(x_axis,Ey[:,ny//2],label='y=0')
ax.plot(y_axis,Ey[nx//2:,],label='x=0')
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_title('Initial Ey')
plt.savefig(os.path.join(working_dir,'Ey_0000_centerline.png'))
plt.close(fig)
plt.clf()
exit(0)
fig,ax = plt.subplots()
pcm=ax.pcolormesh(x_axis,yb_axis,Ey.T,cmap='RdBu')
ax.set_aspect('equal')
ax.set_xlabel('x(m)')
ax.set_ylabel('y(m)')
ax.set_title('Initial Ey')
plt.colorbar(pcm).ax.set_ylabel('Ey(V/m)')
plt.savefig(os.path.join(working_dir,'Ey_0000.png'))
plt.close(fig)
plt.clf()

