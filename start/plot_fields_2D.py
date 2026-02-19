import sdf_helper
from joblib import Parallel, delayed
import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
from scipy.ndimage import zoom
import xarray as xr

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=20/2D/K=-0.005,D=0.02,L=0.00'
zoom_factor=0.1

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2

def plot_2D(sdf_name,Block_name_list=[]):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,sdf_name))
    pprint.pprint(sdf.__dict__)
    time=sdf.__dict__['Header']['time']
    axis_x=sdf.__dict__['Grid_Grid_mid'].data[0]
    axis_y=sdf.__dict__['Grid_Grid_mid'].data[1]
    axis_x=zoom(input=axis_x,zoom=zoom_factor,order=1)
    axis_y=zoom(input=axis_y,zoom=zoom_factor,order=1)
    for Block in Block_name_list:
        Block_name=Block['name']
        Block_data=sdf.__dict__[Block_name].data
        Block_data=zoom(input=Block_data,zoom=zoom_factor,order=1)
        Block_max=np.max(np.abs(Block_data))
        fig,ax = plt.subplots()
        pcm=ax.pcolormesh(axis_x/laser_lambda,axis_y/laser_lambda,Block_data.transpose(1,0)/Block['normalize'],
                          cmap=Block['cmap'],
                          #vmin=-25,vmax=25,
                          #vmin=-Block_max/Block['normalize'],vmax=Block_max/Block['normalize'],
                          )
        #ax.set_xlim(-25,25)
        #ax.set_ylim(-25,25)
        ax.set_aspect('equal')
        ax.set_xlabel('x/λ0')
        ax.set_ylabel('y/λ0')
        ax.set_title('%s at %.2fT0' %(Block_name,time/laser_period))
        plt.colorbar(pcm).ax.set_ylabel(Block['label'])
        plt.savefig(os.path.join(working_dir,'%s_%s.png' %(sdf_name,Block_name)))
        plt.close(fig)
        plt.clf()
Block_name_list=[
    {'name':'Derived_Number_Density_Electron','cmap':'Reds','normalize':laser_Nc,'label':'Ne/Nc'},
    #{'name':'Electric_Field_Ex','cmap':'RdBu','normalize':1,'label':'a=Ex/Ec'},
    {'name':'Electric_Field_Ey','cmap':'RdBu','normalize':laser_Ec,'label':'a=Ey/Ec'},
    {'name':'Magnetic_Field_Bz','cmap':'RdBu','normalize':laser_Bc,'label':'a=Bz/Bc'}
] 

def do_each(sdf_name):
    plot_2D(sdf_name,Block_name_list)

sdf_name_list=[sdf_name for sdf_name in os.listdir(working_dir) if sdf_name.endswith('sdf') and sdf_name.startswith('fields')]
print(sdf_name_list)
Parallel(n_jobs=1)(delayed(do_each)(sdf_name) for sdf_name in sdf_name_list)
exit(0)

for sdf_name in os.listdir(working_dir):
    if sdf_name.endswith('sdf') and sdf_name.startswith('fields'):
        plot_2D(sdf_name,Block_name_list)
