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
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thick/1D'
zoom_factor=1

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
        ax.plot(axis_x,Block_data)
        ax.set_xlabel('x(m)')
        ax.set_ylabel(Block['label'])
        ax.set_title('%s at %es' %(Block_name,time))
        plt.savefig(os.path.join(working_dir,'%s_%0.4d.png' %(Block_name,i)))
        plt.close(fig)
        plt.clf()
Block_name_list=[
    {'name':'Derived_Number_Density_Electron','label':'Ne(1/m^3)'},
    #{'name':'Derived_Number_Density_Ion','label':'Ne(1/m^3)'},
    {'name':'Electric_Field_Ey','label':'Ey(V/m)'},
    #{'name':'Electric_Field_Ez','label':'Ez(V/m)'},
    #{'name':'Magnetic_Field_By','label':'Bt(T)'},
    #{'name':'Magnetic_Field_Bz','label':'Bz(T)'},
] 

for i in np.arange(300):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    plot_1D(sdf,Block_name_list)
