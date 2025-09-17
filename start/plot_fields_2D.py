import sdf_helper
import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
from scipy.ndimage import zoom
import xarray as xr

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/no_curve/45/2D/ND_a0=1.00'
zoom_factor=0.1

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2

def plot_2D(sdf,Block_name_list=[]):
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
        fig,ax = plt.subplots()
        pcm=ax.pcolormesh(axis_x/laser_lambda,axis_y/laser_lambda,Block_data.transpose(1,0)/Block['normalize'],cmap=Block['cmap'])
        ax.set_aspect('equal')
        ax.set_xlabel('x/λ0')
        ax.set_ylabel('y/λ0')
        ax.set_title('%s at %.2fT0' %(Block_name,time/laser_period))
        plt.colorbar(pcm).ax.set_ylabel(Block['label'])
        plt.savefig(os.path.join(working_dir,'%s_%0.4d.png' %(Block_name,i)))
        plt.close(fig)
        plt.clf()
Block_name_list=[
    {'name':'Derived_Number_Density_Electron','cmap':'Reds','normalize':laser_Nc,'label':'Ne/Nc'},
    {'name':'Electric_Field_Ey','cmap':'RdBu','normalize':laser_Ec,'label':'a=Ey/Ec'},
    #{'name':'Electric_Field_Ex','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
    #{'name':'Electric_Field_Ez','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
] 

for i in range(20):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    plot_2D(sdf,Block_name_list)


        

