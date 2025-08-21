import scipy.constants as C
import numpy as np
import os
import math
import meep as mp
import xarray as xr
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/2D_gaussian/meep'
os.makedirs(name=working_dir,exist_ok=True)
os.chdir(path=working_dir)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 3		# Laser field strength
laser_w0=2*laser_lambda
laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_polarisation=0   #rad

cell_x=100*laser_lambda
cell_y=40*laser_lambda

pml_thickness=2*laser_lambda
pml_layers = [mp.PML(thickness=pml_thickness)]
cell_size = mp.Vector3(cell_x,cell_y,0)

cells_per_lambda =200
resolution=cells_per_lambda*(1/laser_lambda)
sim_time=100*laser_lambda

source_x = -0.5*cell_x + pml_thickness + laser_lambda
sources = [
    mp.GaussianBeam2DSource(
        src=mp.ContinuousSource(wavelength=laser_lambda,width=0),
        center=mp.Vector3(source_x,0,0),
        size=mp.Vector3(0,cell_y,0),
        beam_x0=mp.Vector3(-source_x,0,0),
        beam_kdir=mp.Vector3(1,0,0),
        beam_w0=laser_w0,
        beam_E0=mp.Vector3(0,np.cos(laser_polarisation),np.sin(laser_polarisation)),
        amplitude=10,
    )
]
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    dimensions=2,
    #k_point=mp.Vector3(0,  0,0)
)

sim.init_sim()

x_axis=np.linspace(start=sim.fields.gv.xmin(),stop=sim.fields.gv.xmax(),num=sim.fields.gv.nx(),endpoint=False)+sim.fields.gv.dx().x()/2
y_axis=np.linspace(start=sim.fields.gv.ymin(),stop=sim.fields.gv.ymax(),num=sim.fields.gv.ny(),endpoint=False)+sim.fields.gv.dy().y()/2

i=0
def record_fields(sim:mp.Simulation):
    global i
    i=i+1
    Ex=sim.get_efield_x()
    Ey=sim.get_efield_y()
    Ez=sim.get_efield_z()
    timestep=sim.timestep()
    time=sim.meep_time()
    fields=xr.Dataset(
        data_vars={
            'Ex':(["x", "y"], Ex),
            'Ey':(["x", "y"], Ey),
            'Ez':(["x", "y"], Ez),
            },
        coords={'x':(["x"], x_axis),'y':(["y"], y_axis),'timestep':timestep,'time':time}
        )
    nc=os.path.join(working_dir,'fields_%0.4d.nc' %(i))
    fields.to_netcdf(path=nc)
    print(nc)
    with open(os.path.join(working_dir,'output_fields.txt'),mode='a+') as txt:
        txt.write(nc+'\n')


sim.run(mp.at_every(laser_lambda,record_fields), until=sim_time)

