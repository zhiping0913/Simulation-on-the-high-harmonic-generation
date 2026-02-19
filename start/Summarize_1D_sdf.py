import numpy as np
import sdf_helper
import pandas as pd
import xarray as xr
import h5py
import os
import pprint
from read_write import read_sdf,write_fields_to_nc

block_info_list=[
    #{'name':'Electric_Field_Ex','units':'V/m','long_name':'Electric Field Ex in Moving Frame'},
    {'name':'Electric_Field_Ey','units':'V/m','long_name':'Electric Field Ey in Moving Frame'},
    #{'name':'Electric_Field_Ez','units':'V/m','long_name':'Electric Field Ez in Moving Frame'},
    #{'name':'Magnetic_Field_Bx','units':'T','long_name':'Magnetic Field Bx in Moving Frame'},
    #{'name':'Magnetic_Field_By','units':'T','long_name':'Magnetic Field By in Moving Frame'},
    #{'name':'Magnetic_Field_Bz','units':'T','long_name':'Magnetic Field Bz in Moving Frame'},
    #{'name':'Derived_Number_Density_Electron','units':'m^-3','long_name':'Derived Number Density of Electron in Moving Frame'},
    #{'name':'Derived_Number_Density_Ion','units':'m^-3','long_name':'Derived Number Density of Ion in Moving Frame'},
    #{'name':'Derived_Jx_Electron','units':'A/m^2','long_name':'Derived Current Density Jx of Electron in Moving Frame'},
    #{'name':'Derived_Jy_Electron','units':'A/m^2','long_name':'Derived Current Density Jy of Electron in Moving Frame'},
]
block_name_list=[block_info['name'] for block_info in block_info_list]



Data_dict={
    'time':[],
    #'Electron_Gamma':[],
    #'Electron_Vx':[],
    #'Electron_Vy':[],
    #'Ion_Vx':[],
    #'Ion_Vy':[],
    #'Electron_x':[],
    'Electric_Field_Ey':[],
    'Magnetic_Field_Bz':[],
    'Derived_Number_Density_Electron':[],
    #'Derived_Number_Density_Ion':[],
    'Derived_Jx_Electron':[],
    'Derived_Jy_Electron':[],
    'Derived_Jz_Electron':[],

}

i_start=1
i_end=2
n_sdf=i_end-i_start+1

data_dict_list=[]
time_coordinate=[]

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Small_a0/test/with_all_collision/fine'
for i in np.linspace(start=i_start,stop=i_end,num=n_sdf,endpoint=True,dtype=int):
    data_dict=read_sdf(
        sdf_name=os.path.join(working_dir,'%0.4d.sdf' %(i)),
        block_name_list=block_name_list,
    )
    data_dict_list.append(data_dict)
    time_coordinate.append(data_dict['time'])
x_coordinate=data_dict_list[0]['x']
coordinate_dict_list=[
    {'name':'time','coordinate':time_coordinate,'units':'s','long_name':'Time in Moving Frame'},
    {'name':'x','coordinate':x_coordinate,'units':'m','long_name':'X Coordinate in Moving Frame'},
]
df=pd.DataFrame(data=data_dict_list,index=time_coordinate)
field_dict_list=[]
for block_info in block_info_list:
    field_dict_list.append(
        {
            'name':block_info['name'],
            'data':np.stack(df[block_info['name']].to_numpy(),axis=0),
            'units':block_info['units'],
            'long_name':block_info['long_name']
        }
    )
write_fields_to_nc(
    working_dir=working_dir,
    nc_name='Summarize_Field',
    coordinate_dict_list=coordinate_dict_list,
    field_dict_list=field_dict_list
)
exit(0)


for i in np.linspace(start=i_start,stop=i_end,num=n_sdf,endpoint=True,dtype=int):
    data_dict=read_sdf(
        fname=os.path.join(working_dir,'%0.4d.sdf' %(i)),
        block_name_list=block_name_list,
    )
    
    
    
    sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
    time=sdf.__dict__['Header']['time']
    Data_dict['time'].append(time)
    #Electron_ID=sdf.__dict__['Particles_ID_Electron'].data
    #Ion_ID=sdf.__dict__['Particles_ID_Ion'].data
    Grid_Grid_mid=sdf.__dict__['Grid_Grid_mid'].data[0]
    #Data_dict['Electron_Gamma'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Gamma_Electron'].data],columns=Electron_ID,index=[time]))
    #Data_dict['Electron_Vx'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vx_Electron'].data],columns=Electron_ID,index=[time]))
    #Data_dict['Electron_Vy'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vy_Electron'].data],columns=Electron_ID,index=[time]))
    #Data_dict['Ion_Vx'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vx_Ion'].data],columns=Ion_ID,index=[time]))
    #Data_dict['Ion_Vy'].append(pd.DataFrame(data=[sdf.__dict__['Particles_Vy_Ion'].data],columns=Ion_ID,index=[time]))
    #Data_dict['Electron_x'].append(pd.DataFrame(data=[sdf.__dict__['Grid_Particles_Electron'].data[0]],columns=Electron_ID,index=[time]))
    Data_dict['Derived_Number_Density_Electron'].append(sdf.__dict__['Derived_Number_Density_Electron'].data)
    Data_dict['Derived_Jx_Electron'].append(sdf.__dict__['Derived_Jx_Electron'].data)
    Data_dict['Derived_Jy_Electron'].append(sdf.__dict__['Derived_Jy_Electron'].data)
    Data_dict['Derived_Jz_Electron'].append(sdf.__dict__['Derived_Jz_Electron'].data)
    #Data_dict['Derived_Number_Density_Ion'].append(pd.DataFrame(data=[sdf.__dict__['Derived_Number_Density_Ion'].data],index=[time]))
    Data_dict['Electric_Field_Ey'].append(sdf.__dict__['Electric_Field_Ey'].data)
    Data_dict['Magnetic_Field_Bz'].append(sdf.__dict__['Magnetic_Field_Bz'].data)

Data_df_dict={
    #'Electron_Gamma':pd.concat(objs=Data_dict['Electron_Gamma'],axis=0),
    #'Electron_Vx':pd.concat(objs=Data_dict['Electron_Vx'],axis=0),
    #'Electron_Vy':pd.concat(objs=Data_dict['Electron_Vy'],axis=0),
    #'Ion_Vx':pd.concat(objs=Data_dict['Ion_Vx'],axis=0),
    #'Ion_Vy':pd.concat(objs=Data_dict['Ion_Vy'],axis=0),
    #'Electron_x':pd.concat(objs=Data_dict['Electron_x'],axis=0),
    'Electric_Field_Ey':pd.DataFrame(data=Data_dict['Electric_Field_Ey'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    'Magnetic_Field_Bz':pd.DataFrame(data=Data_dict['Magnetic_Field_Bz'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    'Derived_Number_Density_Electron':pd.DataFrame(data=Data_dict['Derived_Number_Density_Electron'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    'Derived_Jx_Electron':pd.DataFrame(data=Data_dict['Derived_Jx_Electron'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    'Derived_Jy_Electron':pd.DataFrame(data=Data_dict['Derived_Jy_Electron'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    'Derived_Jz_Electron':pd.DataFrame(data=Data_dict['Derived_Jz_Electron'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
    #'Derived_Number_Density_Ion':pd.DataFrame(data=Data_dict['Derived_Number_Density_Ion'],index=Data_dict['time'],columns=Grid_Grid_mid,dtype=np.float64),
}
#Data_df_dict['Electron_Gamma'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Gamma',mode='a')
#Data_df_dict['Electron_Vx'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Vx',mode='a')
#Data_df_dict['Electron_Vy'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_Vy',mode='a')
#Data_df_dict['Ion_Vx'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Ion_Vx',mode='a')
#Data_df_dict['Ion_Vy'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Ion_Vy',mode='a')
#Data_df_dict['Electron_x'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Electron.hdf5'),key='Electron_x',mode='a')

Data_df_dict['Electric_Field_Ey'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Electric_Field_Ey',mode='a')
Data_df_dict['Magnetic_Field_Bz'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Magnetic_Field_Bz',mode='a')
Data_df_dict['Derived_Number_Density_Electron'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Number_Density_Electron',mode='a')
Data_df_dict['Derived_Jx_Electron'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Jx_Electron',mode='a')
Data_df_dict['Derived_Jy_Electron'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Jy_Electron',mode='a')
Data_df_dict['Derived_Jz_Electron'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Jz_Electron',mode='a')
#Data_df_dict['Derived_Number_Density_Ion'].to_hdf(path_or_buf=os.path.join(working_dir,'Summarize_Field.hdf5'),key='Derived_Number_Density_Ion',mode='a')






