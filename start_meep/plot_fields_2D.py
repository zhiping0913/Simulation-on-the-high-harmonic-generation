import pprint
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom
import xarray as xr
zoom_factor=0.25

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thin/2D/rotate_field'

def plot_2D(fields:xr.Dataset):
    axis_x=fields['x'].to_numpy()
    axis_y=fields['y'].to_numpy()
    axis_x=zoom(input=axis_x,zoom=zoom_factor,order=1)
    axis_y=zoom(input=axis_y,zoom=zoom_factor,order=1)
    time=fields['time']
    for Block in Block_name_list:
        Block_name=Block['name']
        Block_data=fields[Block_name].to_numpy().transpose(1,0)
        print(np.max(np.abs(Block_data)))
        Block_data=zoom(input=Block_data,zoom=zoom_factor,order=1)
        fig,ax = plt.subplots()
        pcm=ax.pcolormesh(axis_x,axis_y,Block_data,cmap=Block['cmap'])
        ax.set_aspect('equal')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_title('%s at %fs' %(Block_name,time))
        plt.colorbar(pcm).ax.set_ylabel(Block['label'])
        plt.savefig(os.path.join(working_dir,'%s_%s.png' %(filename,Block_name)))
        plt.close(fig)
        plt.clf()
    


Block_name_list=[
    {'name':'Ey','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
    {'name':'Ex','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
    {'name':'Ez','cmap':'RdBu','normalize':1,'label':'Ey(V/m)'},
] 

for filename in os.listdir(working_dir):
    if filename.endswith('nc'):
        fields=xr.open_dataset(os.path.join(working_dir,filename))
        plot_2D(fields=fields)
        print('Plot %s' %(filename))

