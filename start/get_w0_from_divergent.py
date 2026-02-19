import os
import numpy as np
import scipy.constants as C
import shutil
import pandas as pd
import matplotlib.pyplot as plt
theta_degree=45

laser_lambda = 0.8*C.micron		# Laser wavelength, unit:m
laser_f0=1/laser_lambda   #unit: m^-1
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 10		# Laser field strength
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)   #unit: T. 1.338718e+04T for 800nm laser
laser_Ec=laser_Bc*C.speed_of_light   #unit: V/m. 4.013376e+12V/m for 800nm laser
laser_amp=laser_a0*laser_Ec
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_w0_lambda= 3
laser_zR_lambda=C.pi*laser_w0_lambda**2
laser_w0=laser_w0_lambda*laser_lambda
laser_zR=laser_zR_lambda*laser_lambda
laser_theta0=1/(C.pi*laser_w0_lambda)
index=np.array([1,2,3,4,5,6,10])
ND_a0=1.0
index=['1','2','3','4','5','6','10']
Kappa_axis=[-0.05,-0.02,0,0.02,0.05]

data=[]
for Kappa in Kappa_axis:
    working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=10/2D/','%d' %(theta_degree),'ND_a0_%3.2f_Kappa_%+5.3f' %(ND_a0,Kappa))
    divergent_data=pd.read_hdf(os.path.join(working_dir,'divergent_angle.hdf5'),key='reflection')
    #w0=laser_lambda/(C.pi*index*divergent_data['divergent_angle'])
    #divergent_data['waist']=w0
    #divergent_data.to_hdf(os.path.join(working_dir,'divergent_angle.hdf5'),key='reflection')
    data.append(divergent_data['waist'].to_numpy())
data=np.array(data)
print(data)


for i,order in enumerate(index):
    plt.plot(Kappa_axis,laser_lambda/data[:,i],label=f'n={order}')
plt.xlabel('Κ')
plt.ylabel(r'$\frac{\lambda_0}{w_0n}$',rotation=0)
plt.legend()
plt.savefig('/scratch/gpfs/MIKHAILOVA/zl8336/try07.py.png')