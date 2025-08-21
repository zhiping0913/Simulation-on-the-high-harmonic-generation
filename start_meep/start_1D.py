import scipy.constants as C
import numpy as np
import os
import math
import meep as mp
import xarray as xr
from scipy.interpolate import interp1d
from meep.materials import SiO2_aniso
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/try_meep'
os.makedirs(name=working_dir,exist_ok=True)
os.chdir(path=working_dir)

def read_field(nc_name='',name=''):
    field_ds=xr.open_dataset(nc_name)
    field_interpolate=interp1d(x=field_ds["x"],y=field_ds[name], kind='cubic',bounds_error=False, fill_value=0)
    return field_interpolate
    
Ey_interpolator=read_field(nc_name=os.path.join(working_dir,'E_y.nc'),name='E_y')   #unit: V/m
Bz_interpolator=read_field(nc_name=os.path.join(working_dir,'B_z.nc'),name='B_z')   #unit: T

def initialize_Ex(pos:mp.vector3):
    return Ey_interpolator(pos.z)

def initialize_By(pos:mp.vector3):
    return Bz_interpolator(pos.z)*C.speed_of_light

laser_lambda = 0.8*C.micro		# Laser wavelength. unit: m
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 3		# Laser field strength
laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity. unit: s
laser_polarisation=0   #rad



vacuum_length_z_lambda=20   #lambda
cells_per_lambda_z =400

cell_z=2*vacuum_length_z_lambda*laser_lambda
pml_thickness=1*laser_lambda
pml_layers = [mp.PML(thickness=pml_thickness)]


resolution=cells_per_lambda_z*(1/laser_lambda)
Courant=0.3
sim_time=30*laser_lambda

cell_size = mp.Vector3(0, 0, cell_z)  
pml_thickness = laser_lambda

# 等离子体参数 (单位: Meep归一化单位)
plasma_lc=laser_lambda*0.5   #plasma critical wavelength
plasma_kc=2*C.pi/plasma_lc   #plasma frequency ωp   unit: 1/m

def transversal_E_field(pos:mp.vector3):
    field=np.cos(laser_k0*pos.z)*np.exp(-np.square(pos.z/(laser_FWHM*C.speed_of_light)))
    return field

gamma = 0.01   # 阻尼系数 (碰撞频率)

# 创建等离子体材料 (Drude模型)
plasma_medium = mp.Medium(
    epsilon=1.0,  # 背景介电常数
    E_susceptibilities=[
        mp.DrudeSusceptibility(
            frequency=1,
            #gamma=gamma,    # 阻尼系数
            sigma=(1/plasma_lc)**2
        )
    ]
)

# 创建几何结构 - 中心放置等离子体平板
plasma_length = 4.0*laser_lambda  # 等离子体区域长度 lambda
geometry = [mp.Block(
    center=mp.Vector3(0,0,0),
    size=mp.Vector3( mp.inf, mp.inf, plasma_length),
    material=plasma_medium,
)]




# 创建模拟对象
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    resolution=resolution,
    boundary_layers=[mp.PML(pml_thickness)],
    dimensions=1,  # 一维模拟
    Courant=Courant
)
sim.init_sim()
#sim.fields.initialize_field(mp.Ex, initialize_Ex)
sim.fields.initialize_field(mp.Dx, initialize_Ex)
sim.fields.initialize_field(mp.By, initialize_By)
#sim.fields.initialize_field(mp.Hy, initialize_By)

z_axis=np.linspace(start=sim.fields.gv.zmin(),stop=sim.fields.gv.zmax(),num=sim.fields.gv.nz(),endpoint=False)+sim.fields.gv.dz().z()/2
print(np.max(z_axis))
print(len(z_axis))
i=0

def record_fields(sim:mp.Simulation):
    global i
    Ex=sim.get_efield_x()
    Ey=sim.get_efield_y()
    Ez=sim.get_efield_z()
    Bx=sim.get_bfield_x()
    By=sim.get_bfield_y()
    Bz=sim.get_bfield_z()
    Dx=sim.get_dfield_x()
    Hy=sim.get_hfield_y()
    timestep=sim.timestep()
    time=sim.meep_time()
    fields=xr.Dataset(
        data_vars={
            'Ex':(["z"], Ex),
            #'Ey':(["z"], Ey),
            #'Ez':(["z"], Ez),
            #'Bx':(["z"], Bx),
            'By':(["z"], By),
            #'Bz':(["z"], Bz),
            #'Dx':(["z"], Dx),
            #'Hy':(["z"], Hy),
            },
        coords={'z':(["z"], z_axis),'timestep':timestep,'time':time}
        )
    nc=os.path.join(working_dir,'lc=0.5','fields_%0.4d.nc' %(i))
    fields.to_netcdf(path=nc)
    print(nc)
    with open(os.path.join(working_dir,'lc=0.5','output_fields.txt'),mode='a+') as txt:
        txt.write(nc+'\n')
    i=i+1
record_fields(sim)

sim.run(mp.at_every(laser_lambda,record_fields), until=sim_time)
