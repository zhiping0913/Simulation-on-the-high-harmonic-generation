import sdf_helper
import pprint
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Try_3D/dat'


zoom_factor=1

def plot_3D(sdf,Block_name_list=[]):
    pprint.pprint(sdf.__dict__)
    time=sdf.__dict__['Header']['time']
    axis_x=sdf.__dict__['Grid_Grid_mid'].data[0]
    axis_y=sdf.__dict__['Grid_Grid_mid'].data[1]
    axis_z=sdf.__dict__['Grid_Grid_mid'].data[2]
    axis_x=zoom(input=axis_x,zoom=zoom_factor,order=1)
    axis_y=zoom(input=axis_y,zoom=zoom_factor,order=1)
    axis_z=zoom(input=axis_z,zoom=zoom_factor,order=1)
    nx=len(axis_x)
    ny=len(axis_y)
    nz=len(axis_z)
    for Block in Block_name_list:
        Block_name=Block['name']
        Block_data=sdf.__dict__[Block_name].data/Block['normalize']
        Block_data=zoom(input=Block_data,zoom=zoom_factor,order=1)
        fig,ax = plt.subplots()
        pcm1=ax.pcolormesh(axis_x,axis_y,Block_data[:,:,nz//2].T,cmap=Block['cmap'])
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('x-y %s at %fs' %(Block_name,time))
        plt.colorbar(pcm1).ax.set_ylabel(Block['label'])
        plt.savefig(os.path.join(working_dir,'%s_xy_%0.4d.png' %(Block_name,i)))
        plt.close(fig)
        plt.clf()
        fig,ax = plt.subplots()
        pcm2=ax.pcolormesh(axis_x,axis_z,Block_data[:,ny//2,:].T,cmap=Block['cmap'])
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('x-z %s at %fs' %(Block_name,time))
        plt.colorbar(pcm2).ax.set_ylabel(Block['label'])
        plt.savefig(os.path.join(working_dir,'%s_xz_%0.4d.png' %(Block_name,i)))
        plt.close(fig)
        plt.clf()


Block_name_list=[
    #{'name':'Derived_Number_Density_Electron','cmap':'Reds','normalize':1,'label':'Ne(1/m^3)'},
    {'name':'Electric_Field_Ex','cmap':'RdBu','normalize':1,'label':'Ex(V/m)'},
    {'name':'Electric_Field_Ey','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
    {'name':'Electric_Field_Ez','cmap':'RdBu','normalize':1,'label':'Ez(V/m)'},
] 

for i in range(100):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    plot_3D(sdf,Block_name_list)

