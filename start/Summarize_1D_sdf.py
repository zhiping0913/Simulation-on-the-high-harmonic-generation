import numpy as np
import sdf_helper
import pandas as pd
import h5py
import os
import pprint

Data_dict={
    #'Electron_Gamma':[],
    'Electron_Vx':[],
    'Electron_Vy':[],
    'Ion_Vx':[],
    'Ion_Vy':[],
    #'Electron_x':[],
    #'Electric_Field_Ey':[],
    #'Magnetic_Field_Bz':[],
    #'Derived_Number_Density_Electron':[],
    #'Derived_Number_Density_Ion':[],
}

i_max=15
n_sdf=i_max+1

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/try_1D'

for i in range(n_sdf):
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    time=sdf.__dict__['Header']['time']
    Electron_ID=sdf.__dict__['Particles_ID_Electron'].data
    Ion_ID=sdf.__dict__['Particles_ID_Ion'].data
    #Grid_Grid_mid=sdf.__dict__['Grid_Grid_mid'].data[0]
    #Data_dict['Electron_Gamma'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Gamma_Electron'].data],columns=Electron_ID,index=[time]))
    Data_dict['Electron_Vx'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vx_Electron'].data],columns=Electron_ID,index=[time]))
    Data_dict['Electron_Vy'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vy_Electron'].data],columns=Electron_ID,index=[time]))
    Data_dict['Ion_Vx'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vx_Ion'].data],columns=Ion_ID,index=[time]))
    Data_dict['Ion_Vy'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vy_Ion'].data],columns=Ion_ID,index=[time]))
    #Data_dict['Electron_x'].append(pd.DataFrame(data=[sdf.__dict__['Grid_Particles_Electron'].data[0]],columns=Electron_ID,index=[time]))
    #Data_dict['Derived_Number_Density_Electron'].append(pd.DataFrame(data=[sdf.__dict__['Derived_Number_Density_Electron'].data],index=[time]))
    #Data_dict['Derived_Number_Density_Ion'].append(pd.DataFrame(data=[sdf.__dict__['Derived_Number_Density_Ion'].data],index=[time]))
    #Data_dict['Electric_Field_Ey'].append(pd.DataFrame(data=[sdf.__dict__['Electric_Field_Ey'].data],index=[time]))
    #Data_dict['Magnetic_Field_Bz'].append(pd.DataFrame(data=[sdf.__dict__['Magnetic_Field_Bz'].data],index=[time]))

Data_df_dict={
    #'Electron_Gamma':pd.concat(objs=Data_dict['Electron_Gamma'],axis=0),
    'Electron_Vx':pd.concat(objs=Data_dict['Electron_Vx'],axis=0),
    'Electron_Vy':pd.concat(objs=Data_dict['Electron_Vy'],axis=0),
    'Ion_Vx':pd.concat(objs=Data_dict['Ion_Vx'],axis=0),
    'Ion_Vy':pd.concat(objs=Data_dict['Ion_Vy'],axis=0),
    #'Electron_x':pd.concat(objs=Data_dict['Electron_x'],axis=0),
    #'Electric_Field_Ey':pd.concat(objs=Data_dict['Electric_Field_Ey'],axis=0),
    #'Magnetic_Field_Bz':pd.concat(objs=Data_dict['Magnetic_Field_Bz'],axis=0),
    #'Derived_Number_Density_Electron':pd.concat(objs=Data_dict['Derived_Number_Density_Electron'],axis=0),
    #'Derived_Number_Density_Ion':pd.concat(objs=Data_dict['Derived_Number_Density_Ion'],axis=0),
}

#Data_df_dict['Electric_Field_Ey'].columns=Grid_Grid_mid
#Data_df_dict['Magnetic_Field_Bz'].columns=Grid_Grid_mid
#Data_df_dict['Derived_Number_Density_Electron'].columns=Grid_Grid_mid
#Data_df_dict['Derived_Number_Density_Ion'].columns=Grid_Grid_mid

#Data_df_dict['Electron_Gamma'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Gamma',mode='a')
Data_df_dict['Electron_Vx'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Vx',mode='a')
Data_df_dict['Electron_Vy'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Vy',mode='a')
Data_df_dict['Ion_Vx'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Ion_Vx',mode='a')
Data_df_dict['Ion_Vy'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Ion_Vy',mode='a')
#Data_df_dict['Electron_x'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_x',mode='a')

#Data_df_dict['Electric_Field_Ey'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Electric_Field_Ey',mode='a')
#Data_df_dict['Magnetic_Field_Bz'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Magnetic_Field_Bz',mode='a')
#Data_df_dict['Derived_Number_Density_Electron'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Number_Density_Electron',mode='a')
#Data_df_dict['Derived_Number_Density_Ion'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Number_Density_Ion',mode='a')






