import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import XKCD_COLORS
import scipy.constants as C
import os
from scipy.optimize import curve_fit
import pandas as pd
color_keys=list(XKCD_COLORS.keys())
laser_lambda = 0.8*C.micron		# Laser wavelength, unit:m
laser_f0=1/laser_lambda   #unit: m^-1
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 10		# Laser field strength
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)   #unit: T
laser_Ec=laser_Bc*C.speed_of_light   #unit: V/m
laser_amp=laser_a0*laser_Ec
laser_w0_lambda= 3
laser_zR_lambda=C.pi*laser_w0_lambda**2
laser_w0=laser_w0_lambda*laser_lambda
laser_zR=laser_zR_lambda*laser_lambda
laser_theta0=1/(C.pi*laser_w0_lambda)

N=350
a0=10
ND_a0=1.0
D=ND_a0*a0/N
Kappa=-0.05
theta_degree=45
working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/a0=10/2D/','%d' %(theta_degree),'ND_a0_%3.2f_Kappa_%+5.3f' %(ND_a0,Kappa))
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Gaussian_beam_pulse'
print(f'working_dir: {working_dir}')

def theoretical_w_z(z,w0=laser_w0,z_focus=0,k0=laser_k0):
    zR = k0 * w0**2 /2
    return w0 * np.sqrt(1 + np.square((z-z_focus)/zR))

def theoretical_envelope_z(z,w0=laser_w0,z_focus=0,k0=laser_k0,amp=laser_amp):
    zR = k0 * w0**2 /2
    return amp*np.power(1+np.square((z-z_focus)/zR),-1/4)

def fit_w_z(x,w_x,k0=laser_k0,name=''):
    def local_f(z,w0,z_focus):
        return theoretical_w_z(z=z,w0=w0,z_focus=z_focus,k0=k0)
    popt, pcov = curve_fit(local_f,x,w_x,p0=(np.min(w_x),np.average(x)))
    print(popt)
    print(pcov)
    w_z_fit=local_f(x,*popt)
    plt.scatter(x/laser_lambda, w_x/laser_lambda)
    plt.plot(x/laser_lambda, w_z_fit/laser_lambda, label='Fit', color='red',linestyle='--')
    plt.xlabel('x/λ0')
    plt.ylabel('w_z/λ0')
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(working_dir,f'fit_w_z_{name}.png'))
    plt.clf()
    return popt

def fit_envelope_z(x,envelope_x,k0=laser_k0,name=''):
    def local_f(z,w0,z_focus,amp):
        return theoretical_envelope_z(z=z,w0=w0,z_focus=z_focus,k0=k0,amp=amp)
    popt, pcov = curve_fit(local_f,x,envelope_x,p0=(laser_w0,np.average(x),np.max(envelope_x)))
    print(popt)
    print(pcov)
    envelope_x_fit=local_f(x,*popt)
    plt.scatter(x/laser_lambda, envelope_x/laser_Ec)
    plt.plot(x/laser_lambda, envelope_x_fit/laser_Ec, label='Fit', color='red',linestyle='--')
    plt.xlabel('x/λ0')
    plt.ylabel('E/Ec')
    plt.legend()
    plt.title(name)
    plt.savefig(os.path.join(working_dir,f'fit_envelope_z_{name}.png'))
    plt.clf()
    return popt

data=pd.read_hdf(path_or_buf=os.path.join(working_dir,'waist.hdf5'),key='Ey')
Field_envelope_moment_1_at_x=data.loc[:,'Field_envelope_moment_1_at_x'].to_numpy()
energy_flux_on_y_width=data.loc[:,'energy_flux_on_y_width'].to_numpy()
Field_envelope_centercross_peak_width=data.loc[:,'Field_envelope_center_y_profile_peak_width'].to_numpy()
Field_envelope_moment_1_at_x=data.loc[:,'Field_envelope_moment_1_at_x'].to_numpy()
Field_envelope_at_moment_1=data.loc[:,'Field_envelope_at_moment_1'].to_numpy()
Field_envelope_max=data.loc[:,'Field_envelope_max'].to_numpy()
Field_envelope_max_at_x=data.loc[:,'Field_envelope_max_at_x'].to_numpy()
fit_w_z_popt=fit_w_z(Field_envelope_max_at_x,energy_flux_on_y_width,k0=laser_k0,name='Ey_flux_on_y_width')
fit_w_z_popt=fit_w_z(Field_envelope_max_at_x,Field_envelope_centercross_peak_width,k0=laser_k0,name='Ey_centercross_peak_width')
fit_envelope_z_popt=fit_envelope_z(Field_envelope_max_at_x,Field_envelope_max,k0=laser_k0,name='Ey')
width=energy_flux_on_y_width
width_derivative=np.gradient(width,Field_envelope_max_at_x)
print(width_derivative[-1])
print(width_derivative.max())
#ax.plot(Field_envelope_moment_1_at_x/laser_lambda,width_derivative,label=key)
exit(0)

order_interested={
    #'all':(0,60),
    '1':(0.5,1.5),
    '2':(1.5,2.5),
    '3':(2.5,3.5),
    '4':(3.5,4.5),
    '5':(4.5,5.5),
    '10':(9.5,10.5),
    #'15':(14.5,15.5),
    #'[9,60]':(8.5,60.5)
}
fig,ax = plt.subplots()
key_list=['1','2','3','4','5','10']
for key in key_list:
    print(f'Processing order: {key}')
    data=pd.read_hdf(path_or_buf=os.path.join(working_dir,'waist_fine.hdf5'),key=f'reflection_{key}')
    Field_envelope_moment_1_at_x=data.loc[:,'Field_envelope_moment_1_at_x'].to_numpy()
    energy_flux_on_y_width=data.loc[:,'energy_flux_on_y_width'].to_numpy()
    width=energy_flux_on_y_width
    width_derivative=np.gradient(width,Field_envelope_moment_1_at_x)
    print(width_derivative[-1])
    print(width_derivative.max())
    #ax.plot(Field_envelope_moment_1_at_x/laser_lambda,width_derivative,label=key)
exit(0)
plt.xlabel('x/λ0')
plt.ylabel('d(waist)/d(x)')
plt.title('Derivative of Waist')
plt.legend()
plt.savefig(os.path.join(working_dir,'derivative_of_waist.png'))
exit(0)
for key in order_interested.keys():
    data=pd.read_hdf(path_or_buf=os.path.join(working_dir,'waist_fine.hdf5'),key=f'reflection_{key}')
    Field_envelope_centercross_peak_width=data.loc[:,'Field_envelope_center_y_profile_peak_width'].to_numpy()
    Field_envelope_moment_1_at_x=data.loc[:,'Field_envelope_moment_1_at_x'].to_numpy()
    Field_envelope_at_moment_1=data.loc[:,'Field_envelope_at_moment_1'].to_numpy()
    Field_envelope_max=data.loc[:,'Field_envelope_max'].to_numpy()
    Field_envelope_max_at_x=data.loc[:,'Field_envelope_max_at_x'].to_numpy()
    Field_envelope_center_y_profile_peak_left=data.loc[:,'Field_envelope_center_y_profile_peak_left'].to_numpy()
    Field_envelope_center_y_profile_peak_right=data.loc[:,'Field_envelope_center_y_profile_peak_right'].to_numpy()
    energy_flux_on_y_width=data.loc[:,'energy_flux_on_y_width'].to_numpy()
    fit_w_z_popt=fit_w_z(Field_envelope_moment_1_at_x,energy_flux_on_y_width,k0=eval(key)*laser_k0,name=f'reflection_{key}')
    fit_envelope_z_popt=fit_envelope_z(Field_envelope_max_at_x,Field_envelope_max,k0=eval(key)*laser_k0,name=f'reflection_{key}')

    pd.DataFrame(data=np.concatenate((fit_w_z_popt,fit_envelope_z_popt)).reshape(1,5),columns=['w0_from_waist','z_focus_from_waist','w0_from_amp','z_focus_from_amp','amp_from_amp']).to_hdf(path_or_buf=os.path.join(working_dir,'w0_fit.hdf5'),key=f'reflection_{key}')



