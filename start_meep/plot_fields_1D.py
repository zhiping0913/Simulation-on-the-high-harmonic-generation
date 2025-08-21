import numpy as np
import scipy.constants as C
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import xarray as xr
zoom_factor=1

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/try_meep/lc=0.5'
laser_lambda = 0.8*C.micron		# Laser wavelength
laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_A0=(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)

def plot_1D(fields:xr.Dataset):
    axis_z=fields['z'].to_numpy()
    axis_z=zoom(input=axis_z,zoom=zoom_factor,order=1)
    time=fields['time']
    for Block in Block_name_list:
        Block_name=Block['name']
        Block_data=fields[Block_name].to_numpy()
        Block_data=zoom(input=Block_data,zoom=zoom_factor,order=1)
        fig,ax = plt.subplots()
        ax.plot(axis_z/laser_lambda,Block_data/laser_A0)
        ax.set_xlabel('z/λ0')
        ax.set_ylabel(Block['label'])
        ax.set_title('%s at %es' %(Block_name,time))
        ax.set_ylim(-0.015,0.015)
        plt.savefig(os.path.join(working_dir,'%s_%s.png' %(filename,Block_name)))
        plt.close(fig)
        plt.clf()
    
def read_nc(nc_name='',key_name_list=[]):
    nc=xr.open_dataset(filename_or_obj=nc_name)
    data_dict={}
    for key_name in key_name_list:
        data_dict[key_name]=nc[key_name].to_numpy()
    return data_dict

Block_name_list=[
    #{'name':'Ey','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
    {'name':'Ex','cmap':'RdBu','normalize':1,'label':'Ex(a0)'},
    #{'name':'Dx','cmap':'RdBu','normalize':1,'label':'Dx(V/m)'},
    #{'name':'Ez','cmap':'RdBu','normalize':1,'label':'Ez(V/m)'},
    #{'name':'By','cmap':'RdBu','normalize':1,'label':'By(V/m)'},
    #{'name':'Hy','cmap':'RdBu','normalize':1,'label':'Hy(V/m)'},
] 

for i in np.linspace(start=0,stop=100,endpoint=False,num=100,dtype=np.int64):
    filename='fields_%0.4d.nc' %(i)
    print(filename)
    fields=xr.open_dataset(filename_or_obj=os.path.join(working_dir,filename))
    plot_1D(fields)


