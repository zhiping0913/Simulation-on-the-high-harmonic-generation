import sdf_helper
import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.ndimage import zoom
import xarray as xr
from start import read_nc
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/no_curve/45/1D/ND_a0=0.25'
zoom_factor=1

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
def plot_1D(sdf,Block_name_list=[]):
    #pprint.pprint(sdf.__dict__)
    time=sdf.__dict__['Header']['time']
    axis_x=sdf.__dict__['Grid_Grid_mid'].data[0]
    axis_x=zoom(input=axis_x,zoom=zoom_factor,order=1)
    for Block in Block_name_list:
        Block_name=Block['name']
        Block_data=sdf.__dict__[Block_name].data
        Block_data=zoom(input=Block_data,zoom=zoom_factor,order=1)
        fig,ax = plt.subplots()
        ax.plot(axis_x/laser_lambda,Block_data/Block['normalize'])
        ax.set_xlabel('x/λ0')
        ax.set_ylabel(Block['label'])
        ax.set_title('%s at %.2fT0' %(Block_name,time/laser_period))
        plt.savefig(os.path.join(working_dir,'%s_%0.4d.png' %(Block_name,i)))
        plt.close(fig)
        plt.clf()
Block_name_list=[
    {'name':'Derived_Number_Density_Electron','label':'Ne/Nc','normalize':laser_Nc},
    #{'name':'Derived_Number_Density_Ion','label':'Ne(1/m^3)'},
    {'name':'Electric_Field_Ey','label':'Ey/Ec','normalize':laser_Ec},
    #{'name':'Electric_Field_Ez','label':'Ez(V/m)'},
    #{'name':'Magnetic_Field_By','label':'Bt(T)'},
    #{'name':'Magnetic_Field_Bz','label':'Bz/Bc','normalize':laser_Bc},
] 

for i in np.arange(300):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    plot_1D(sdf,Block_name_list)
