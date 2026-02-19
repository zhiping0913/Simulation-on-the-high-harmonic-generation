import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import os
import jax.numpy as jnp
from joblib import Parallel, delayed
import math
import scipy.constants as C
import numpy as np
from typing import Union, Tuple, Optional
from Lorentz.Lorentz_transform import LorentzTransform
from start import read_sdf
from plot.plot_1D import plot_multiple_1D_fields

theta_degree=45
theta_rad=jnp.radians(theta_degree)

laser_lambda = 0.875*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0=1		# Laser field strength
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
laser_Ec=laser_Bc*C.speed_of_light   #4.013376e+12V/m
laser_Sc=C.epsilon_0*C.speed_of_light*laser_Ec**2/2   #1.327e+18 W/m^2
laser_amp=laser_a0*laser_Ec
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
#laser_tau=laser_period/jnp.sqrt(jnp.pi)
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_S0=C.epsilon_0*C.speed_of_light*laser_amp**2/2   #average Poynting vector

target_N=200

cells_per_lambda =2000
vacuum_length_x_lambda=20   #lambda
continuation_length_lambda=2000   #lambda
space_length_lambda=vacuum_length_x_lambda+2*continuation_length_lambda   #lambda
n_field_x=round(vacuum_length_x_lambda*cells_per_lambda)
n_continuation_x=round(space_length_lambda*cells_per_lambda)

d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation_x

print('λ0/dx=',laser_lambda/d_x)
print('f0/df=',laser_f0/d_f)

laser_lambda_M=laser_lambda/jnp.cos(theta_rad)
laser_period_M=laser_period/jnp.cos(theta_rad)
laser_tau_M=laser_tau/jnp.cos(theta_rad)
laser_f0_M=laser_f0*jnp.cos(theta_rad)
laser_k0_M=laser_k0*jnp.cos(theta_rad)
laser_Bc_M=laser_Bc*jnp.cos(theta_rad)
laser_Ec_M=laser_Ec*jnp.cos(theta_rad)
laser_Nc_M=laser_Nc/jnp.cos(theta_rad)
laser_Sc_M=laser_Sc*jnp.cos(theta_rad)**2
laser_amp_M=laser_amp*jnp.cos(theta_rad)
laser_S0_M=laser_S0*jnp.cos(theta_rad)**2
vacuum_length_x_lambda_M=vacuum_length_x_lambda*jnp.cos(theta_rad)
space_length_lambda_M=space_length_lambda*jnp.cos(theta_rad)   #laser_f0_M/d_f

laser_spectrum_peak_M=laser_amp_M*(jnp.sqrt(C.pi)/2)*(laser_tau_M*C.speed_of_light)*(1-jnp.exp(-laser_k0_M**2*(laser_tau_M*C.speed_of_light)**2))
laser_energy_M=laser_amp_M**2*jnp.sqrt(C.pi/2)*(laser_tau_M*C.speed_of_light/2)*(1-jnp.exp(-laser_k0_M**2*(laser_tau_M*C.speed_of_light)**2/2))
laser_envelope_integral_M=laser_amp_M**2*(laser_tau_M*C.speed_of_light)*jnp.sqrt(C.pi/2)

plasma_cutoff_order=jnp.sqrt(target_N/(jnp.cos(theta_rad))**3)

highest_harmonic=100000


lorentz_transform = LorentzTransform(
    beta_x=0,
    beta_y=np.sin(np.radians(theta_degree)),
    beta_z=0
    )
print(lorentz_transform)
print(lorentz_transform.transform_matrix)




def transform_current(
    Number_Density_M:jnp.ndarray,
    Jx_M:Optional[jnp.ndarray]=None,
    Jy_M:Optional[jnp.ndarray]=None,
    Jz_M:Optional[jnp.ndarray]=None,
    direction:str='1->0',
    charge=-C.elementary_charge,
    ):
    trans_current=lorentz_transform.transform_field(
        component0=Number_Density_M*charge,  # Convert number density to charge density
        component1=Jx_M if Jx_M is not None else jnp.zeros_like(Number_Density_M),
        component2=Jy_M if Jy_M is not None else jnp.zeros_like(Number_Density_M),
        component3=Jz_M if Jz_M is not None else jnp.zeros_like(Number_Density_M),
        four_vector_type='four_current',
        direction=direction
    )
    Number_Density_L=trans_current[0]/charge
    Jx_L=trans_current[1]
    Jy_L=trans_current[2]
    Jz_L=trans_current[3]

    return Number_Density_L, Jx_L, Jy_L, Jz_L

