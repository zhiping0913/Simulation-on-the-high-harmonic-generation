import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import os
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.constants as C
from scipy.signal import peak_widths, hilbert
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import zoom
from jax.numpy.fft import fftfreq,fftshift,ifftshift,fft2,ifft2
from matplotlib.colors import LogNorm,Normalize
import pandas as pd
import cv2
import xarray as xr
from start import read_sdf,read_nc,read_dat
from pretreat_fields_2D import square_integral_field_2D, continue_field_2D, write_field_2D
D=5
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/45/ND_a0_1.00_Kappa_+0.000'
working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thick/2D'
working_dir=f'/scratch/gpfs/MIKHAILOVA/zl8336/try_evolution/D={D}'
theta_degree=0
theta_rad=jnp.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength, unit:m
laser_f0=1/laser_lambda   #unit: m^-1
laser_k0=2*C.pi*laser_f0
laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 10		# Laser field strength
laser_a0 = 1		# Laser field strength
laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)   #unit: T
laser_Ec=laser_Bc*C.speed_of_light   #unit: V/m
laser_amp=laser_a0*laser_Ec
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_FWHM=2*C.femto   #The full width at half maximum of the intensity.
laser_tau=laser_FWHM/jnp.sqrt(2*jnp.log(2)) 
laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_w0_lambda= D/2
laser_zR_lambda=C.pi*laser_w0_lambda**2
laser_w0=laser_w0_lambda*laser_lambda
laser_zR=laser_zR_lambda*laser_lambda
laser_theta0=1/(C.pi*laser_w0_lambda)
highest_harmonic=5

def theoretical_laser_energy(amp,z_length,w_0):
    return (C.pi/2)*jnp.square(amp)*z_length*w_0/2


laser_spectrum_peak=C.pi*laser_amp*(laser_tau*C.speed_of_light)*(laser_w0_lambda*laser_lambda)/2*jnp.sqrt(jnp.pi/2)
laser_centerline_energy=laser_amp**2*jnp.sqrt(C.pi/2)*(laser_tau*C.speed_of_light/2)
laser_energy=laser_amp**2*jnp.sqrt(C.pi/2)*(laser_tau*C.speed_of_light/2)*D*laser_lambda
laser_spectrum_on_x_peak=jnp.sqrt(C.pi**3/2)*laser_amp**2*(laser_tau*C.speed_of_light)**2*(laser_w0_lambda*laser_lambda)/4*jnp.sqrt(jnp.pi)   #unit: V^2·m
laser_spectrum_on_y_peak=jnp.sqrt(C.pi**3/2)*laser_amp**2*(laser_tau*C.speed_of_light)*(laser_w0_lambda*laser_lambda)**2/2*jnp.sqrt(jnp.pi)   #unit: V^2·m

cells_per_lambda =250
vacuum_length_x_lambda=20   #lambda
vacuum_length_y_lambda=20   #lambda

cells_per_lambda =200
vacuum_length_x_lambda=5   #lambda
vacuum_length_y_lambda=10   #lambda



continuation_length_lambda=50  #lambda
space_length_lambda=2*(max(vacuum_length_x_lambda,vacuum_length_y_lambda)+continuation_length_lambda)
n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda)
n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda)
n_continuation=round(space_length_lambda*cells_per_lambda)
n_freq_radius=round(n_continuation*(highest_harmonic/cells_per_lambda))
n_freq_angle=n_freq_radius*2


d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation. d_f/laser_f0=1/space_length_lambda
d_f_radius=d_f
d_f_angle=2*C.pi/n_freq_angle

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda

grid_x_axis=jnp.linspace(start=x_min,stop=x_max,num=n_field_x,endpoint=False)+d_x/2
grid_y_axis=jnp.linspace(start=y_min,stop=y_max,num=n_field_y,endpoint=False)+d_x/2

freq_x_axis=fftshift(fftfreq(n=n_continuation,d=d_x))
freq_y_axis=freq_x_axis

freq_r_axis=fftfreq(n=n_continuation,d=d_x)[0:n_freq_radius]
freq_a_axis=jnp.linspace(start=0,stop=2*C.pi,num=n_freq_angle,endpoint=False)

freq_a,freq_r=jnp.meshgrid(freq_a_axis,freq_r_axis,indexing='ij')
freq_x,freq_y=jnp.meshgrid(freq_x_axis,freq_y_axis, indexing='ij')   #freq_x.shape=(n_continuation,n_continuation)
freq_z=jnp.zeros(shape=(n_continuation,n_continuation))
freq_radius=jnp.hypot(freq_x,freq_y)   #shape=(n_continuation,n_continuation)
freq_theta = jnp.arctan2(freq_y,freq_x)

freq_vector=jnp.array((freq_x,freq_y,freq_z))   #shape=(3,n_continuation,n_continuation)
freq_radius_vector_nonzero=freq_radius.at[freq_radius<d_f/100].set(d_f/100)   #Avoid divided by zero
freq_unit_vector=freq_vector/freq_radius_vector_nonzero[jnp.newaxis, :, :]    #shape=(3,n_continuation,n_continuation)


grid_center_mask=jnp.s_[round((n_continuation-n_field_x)/2):round((n_continuation+n_field_x)/2),round((n_continuation-n_field_y)/2):round((n_continuation+n_field_y)/2)]
freq_center_mask=jnp.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius,n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]
freq_axis_center_mask=jnp.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]

print('space shape: ',(n_field_x,n_field_y))
print('freq_x.shape:',freq_x.shape)
print('freq_x[freq_center_mask].shape:',freq_x[freq_center_mask].shape)
print('λ0/dx=',laser_lambda/d_x)
print('f0/df=',laser_f0/d_f)

zoom_factor=0.5

def Gaussian(x,x0,w,A):
    return A*jnp.exp(-jnp.square((x-x0)/w))

def theoretical_w_z(z,z_focus,w0):
    global laser_kn
    zR = laser_kn * w0**2 /2
    return w0 * jnp.sqrt(1 + jnp.square((z-z_focus)/zR))

def theoretical_Kappa_z(z,z_focus,w0):
    global laser_kn
    zR = laser_kn * w0**2 /2
    Kappa_Z=(z-z_focus)/(jnp.square(z-z_focus)+jnp.square(zR))
    return Kappa_Z

def gaussian_beam_profile(x_y,A,z_focus,z_center,w0,z_length):
    x,y=x_y
    w_z=theoretical_w_z(z=x,z_focus=z_focus,w0=w0)
    Field_envelope=A*jnp.sqrt(w0/w_z)*jnp.exp(-jnp.square(y/w_z))*jnp.exp(-jnp.square((x-z_center)/z_length))
    return Field_envelope

def w_z_residuals(params, x, y, weights):
    return weights * (y - theoretical_w_z(x, *params))

def theoretical_transversal_field_envelope(x,z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    transversal_field_envelope=jnp.power(1+jnp.square((z-z_focus)/zR),-1/4)*jnp.exp(-jnp.square(x/w_z))
    return transversal_field_envelope

def theoretical_longitudinal_field_envelope(x,z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    longitudinal_field_envelope=jnp.power(1+jnp.square((z-z_focus)/zR),-3/4)*jnp.exp(-jnp.square(x/w_z))*x/zR
    return longitudinal_field_envelope

def Gouy_phase(z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    return jnp.atan((z-z_focus)/zR)

def plot_field(Field:jnp.ndarray,ax=None,grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,threshold=None,normalize=laser_Ec,vmin=-1.1*laser_a0,vmax=1.1*laser_a0,cmap='RdBu',label='a=E/Ec=B/Bc',name=''):
    assert Field.ndim==2
    assert grid_x_axis.ndim==1
    assert grid_y_axis.ndim==1
    assert Field.shape==(grid_x_axis.size,grid_y_axis.size)
    if ax==None:
        return_ax=False
        fig,ax = plt.subplots()
    else:
        return_ax=True
    Field_zoom=zoom(Field,zoom=zoom_factor)/normalize
    if threshold!=None:
        Field_masked = np.ma.masked_where(condition=jnp.abs(Field_zoom) <= threshold, a=Field_zoom)
    else:
        Field_masked=Field_zoom
    norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
    pcm=ax.pcolormesh(zoom(grid_x_axis,zoom=zoom_factor)/laser_lambda,zoom(grid_y_axis,zoom=zoom_factor)/laser_lambda,Field_masked.T,norm=norm,cmap=cmap)
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel(label)
    ax.set_xlabel('x/λ0')
    ax.set_ylabel('y/λ0')
    ax.set_title(name)
    if return_ax:
        return ax
    else:
        plt.savefig(os.path.join(working_dir,'Field_%s.png' %(name)))
        print(os.path.join(working_dir,'Field_%s.png' %(name)))
        plt.close(fig)
        plt.clf()

def plot_2D_spectrum(Field_spectrum_square:jnp.ndarray,freq_x_axis=freq_x_axis,freq_y_axis=freq_y_axis,name=''):
    assert Field_spectrum_square.ndim==2
    assert freq_x_axis.ndim==1
    assert freq_y_axis.ndim==1
    assert Field_spectrum_square.shape==(freq_x_axis.size,freq_y_axis.size)
    norm = LogNorm(vmin=1e-6, vmax=1.2)
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_x_axis,zoom=zoom_factor)/laser_f0,zoom(freq_y_axis,zoom=zoom_factor)/laser_f0,zoom(Field_spectrum_square.T,zoom=zoom_factor)/laser_spectrum_peak**2,cmap='hot', norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('kx/k0')
    ax.set_ylabel('ky/k0')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum I(kx,ky)/I0' %(name))
    plt.savefig(os.path.join(working_dir,'Spectrum_square_%s.png' %(name)))
    print(os.path.join(working_dir,'Spectrum_square_%s.png' %(name)))
    plt.close(fig)
    plt.clf()


def filter_spectrum(Field_spectrum:jnp.ndarray,filter_range=(1.5,highest_harmonic),return_type='spectrum',name=''):
    assert Field_spectrum.shape==(n_continuation,n_continuation)
    assert return_type in ['field','spectrum','both']
    spectrum_mask=(freq_radius*laser_lambda>filter_range[0])&(freq_radius*laser_lambda<filter_range[1])
    Field_filter_spectrum=Field_spectrum*spectrum_mask
    #Field_filter_energy=square_integral_field_2D(Field=Field_filter_spectrum,d_x=d_f,d_y=d_f,complex_array=True)
    #print('total energy: %f theoretical total energy' %(jnp.sum(Field_spectrum_square)*d_f*d_f/laser_energy))
    #print('filter energy: %f theoretical total energy' %(Field_filter_energy/laser_energy))
    if return_type=='spectrum':
        return {'Field_filter_spectrum':Field_filter_spectrum}
    else:
        Field_filter=get_field_from_spectrum(Field_filter_spectrum,name=name)
        plot_field(Field=Field_filter,name=name)
        plt.plot(grid_x_axis/laser_lambda,Field_filter[n_field_y//2]/laser_amp)
        plt.xlabel('x/λ0')
        plt.ylabel('E_filter (E0)')
        #plt.ylim(-0.3,0.3)
        plt.title('centerline of filtered field')
        plt.savefig(os.path.join(working_dir,'Field_filter_centerline_%s.png' %(name)))
        plt.clf()
        return {'Field_filter_spectrum':Field_filter_spectrum,'Field_filter':Field_filter}
        
        

#@profile
def get_envelope(field_or_spectrum:jnp.ndarray,input_type='field',name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
        input_type : {'field', 'spectrum'}, optional
    """
    assert input_type in ['field', 'spectrum']
    if input_type=='field':
        Field=field_or_spectrum
        assert Field.shape==(n_field_x,n_field_y)
        Field_continuation=continue_field_2D(Field,n_continuation_x=n_continuation,n_continuation_y=n_continuation,smooth=False)
        Field_analytic=hilbert(Field_continuation,axis=0)[grid_center_mask]
    else:
        Field_spectrum=field_or_spectrum
        assert Field_spectrum.shape==(n_continuation,n_continuation)
        h=jnp.sign(freq_x_axis)+1   #shape=(n_continuation,)
        Field_spectrum_hilbert=jnp.einsum('ij,i->ij',Field_spectrum,h)
        Field_analytic=ifft2(ifftshift(Field_spectrum_hilbert))[grid_center_mask]*n_continuation*d_f*n_continuation*d_f
    Field_envelope=jnp.abs(Field_analytic)
    Field_envelope_max=jnp.max(Field_envelope).item()
    Field_analytic_phase=jnp.angle(Field_analytic)
    Field_analytic_phase = np.ma.masked_where(condition=jnp.abs(Field_envelope) <= 0.001*Field_envelope_max, a=Field_analytic_phase)
    #plot_field(Field_analytic_phase,normalize=np.pi,vmin=-1,vmax=1,label='phase(π)',cmap='hsv',name='phase')
    
    
    Field_envelope_max_id=(n_field_x//2,n_field_y//2)   #Field_envelope_max_id=(x_id,y_id)
    print('Field_envelope_max: %f×Ec'%(Field_envelope_max/laser_Ec))
    print(Field_envelope_max_id)
    #Field_envelope_max_id=(n_field_x//2,n_field_y//2)
    Field_envelope_max_phase=Field_analytic_phase[Field_envelope_max_id]
    print('Phase at the peak: %fπ'%(Field_envelope_max_phase/C.pi))
    return {
        'Field_envelope':Field_envelope,
        'Field_analytic_phase':Field_analytic_phase,
        'Field_envelope_max':Field_envelope_max,
        'Field_envelope_max_id':Field_envelope_max_id,
        'Field_envelope_max_phase':Field_envelope_max_phase,
    }
    
    Field_analytic_phase_centerline=Field_analytic_phase[:,Field_envelope_max_id[1]]
    Field_analytic_phase_unwrap=jnp.unwrap(Field_analytic_phase,axis=0,period=2*C.pi)
    Field_analytic_phase_unwrap=Field_analytic_phase_unwrap-Field_analytic_phase_unwrap[Field_envelope_max_id]+Field_envelope_max_phase   #Phase relative to the peak. Keep the phase at the peak
    Field_propagate_phase=laser_k0*(grid_x_axis-grid_x_axis[Field_envelope_max_id[0]])+Field_envelope_max_phase
    Field_phase_extra=Field_analytic_phase_unwrap-Field_propagate_phase[:,jnp.newaxis]   # Gouy phase
    Field_phase_gradient=jnp.gradient(Field_phase_extra,grid_x_axis,axis=0)
    zR=-0.5/(jnp.average(Field_phase_gradient[Field_envelope_max_id[0]-5:Field_envelope_max_id[0]+105,Field_envelope_max_id[1]]))
    W0=jnp.sqrt(laser_lambda*zR/C.pi)
    print(zR/laser_lambda)
    print(W0/laser_lambda)
    
    plt.plot(grid_x_axis/laser_lambda,Field_phase_extra[:,Field_envelope_max_id[1]]/C.pi,label='envelope phase',c='b',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda,-0.5*jnp.atan((grid_x_axis-grid_x_axis[Field_envelope_max_id[0]])/zR)/C.pi,label='Gouy phase',linestyle='--',c='g',linewidth=1)
    #plt.xlim(grid_x_axis[Field_envelope_max_id[0]]/laser_lambda-1,grid_x_axis[Field_envelope_max_id[0]]/laser_lambda+1)
    plt.ylim(-1,1)
    plt.xlabel(xlabel='x_L/λ0')
    plt.ylabel(ylabel='phase/π')
    plt.legend()
    plt.title(label='Phase_%s' %(name))
    plt.savefig(os.path.join(working_dir,'Field_phase_extra_%s.png' %(name)))
    plt.clf()


#@profile
def get_envelope_width(field_or_spectrum:jnp.ndarray,input_type='field',grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
        input_type : {'field', 'spectrum'}, optional
    """
    assert input_type in ['field', 'spectrum']
    n_x=grid_x_axis.size
    n_y=grid_y_axis.size
    d_x=grid_x_axis[1]-grid_x_axis[0]
    d_y=grid_y_axis[1]-grid_y_axis[0]
    Field_envelope_dict=get_envelope(field_or_spectrum=field_or_spectrum,input_type=input_type,name=name)
    grid=jnp.asarray(jnp.meshgrid(grid_x_axis,grid_y_axis, indexing='ij'))   #shape=(2,n_field_x,n_field_y)
    Field_envelope=Field_envelope_dict['Field_envelope']   #shape==(n_field_x,n_field_y)
    Field_envelope_max=Field_envelope_dict['Field_envelope_max']
    Field_envelope_max_id=Field_envelope_dict['Field_envelope_max_id']
    Field_envelope_square=jnp.square(Field_envelope)
    Field_envelope_moment_1=np.average(a=grid,axis=(1,2),weights=Field_envelope_square)   #shape=(2,). average position of the envelope (x,y)
    Field_envelope_moment_2=np.average(a=jnp.square(grid),axis=(1,2),weights=Field_envelope_square)   #shape=(2,)
    Field_envelope_moment_std=jnp.sqrt(Field_envelope_moment_2-jnp.square(Field_envelope_moment_1))
    Field_envelope_moment_1_x_id=(jnp.abs(grid_x_axis - Field_envelope_moment_1[0])).argmin().item()
    Field_envelope_moment_1_y_id=(jnp.abs(grid_y_axis - Field_envelope_moment_1[1])).argmin().item()
    #Field_envelope_center_id=Field_envelope_max_id
    Field_envelope_center_id=(n_x//2,n_y//2)
    #Field_envelope_center_id=(Field_envelope_moment_1_x_id,Field_envelope_moment_1_y_id)
    print(Field_envelope_center_id)
    x_center=grid_x_axis[Field_envelope_center_id[0]]
    y_center=grid_y_axis[Field_envelope_center_id[1]]
    #print('FWHM obtained from envelope_x_std: %ffs'%(Field_envelope_x_std*2*jnp.sqrt(2*jnp.log(2))/C.speed_of_light/C.femto))
    Field_envelope_center_y_profile=Field_envelope[Field_envelope_center_id[0],:]
    Field_envelope_center_x_profile=Field_envelope[:,Field_envelope_center_id[1]]
    Field_envelope_at_moment_1=Field_envelope[Field_envelope_center_id].item()
    Field_envelope_center_y_profile_peak=jnp.asarray(peak_widths(x=Field_envelope_center_y_profile,peaks=[jnp.argmax(a=Field_envelope_center_y_profile)],rel_height=1-jnp.exp(-1)),dtype=jnp.float64).flatten()
    print(Field_envelope_center_y_profile_peak)
    Field_envelope_center_x_profile_peak=jnp.asarray(peak_widths(x=Field_envelope_center_x_profile,peaks=[jnp.argmax(a=Field_envelope_center_x_profile)],rel_height=1-jnp.sqrt(2)/2)).flatten()
    Field_envelope_center_y_profile_peak_width=Field_envelope_center_y_profile_peak[0].item()*d_x   #unit: m
    Field_envelope_center_x_profile_peak_width=Field_envelope_center_x_profile_peak[0].item()*d_x   #unit: m
    x_left,x_right=map_coordinates(input=grid_x_axis,coordinates=[Field_envelope_center_x_profile_peak[-2:]],order=1)   #unit: m
    y_left,y_right=map_coordinates(input=grid_y_axis,coordinates=[Field_envelope_center_y_profile_peak[-2:]],order=1)   #unit: m
    Field_envelope_center_y_profile_peak_waist=Field_envelope_center_y_profile_peak_width/2
    Field_envelope_center_x_profile_FWHM=Field_envelope_center_x_profile_peak_width/C.speed_of_light   #unit: s

    print('waist at the peak: %fλ0' %(Field_envelope_center_y_profile_peak_waist/laser_lambda))
    print('centerline FWHM: %ffs' %(Field_envelope_center_x_profile_FWHM/C.femto))
    print('Field_envelope_center_y_profile_peak,(left,right): (%f,%f)λ0'%(y_left/laser_lambda,y_right/laser_lambda))


    #if input_type=='spectrum':
    #    pass
    #    #Field=get_field_from_spectrum(field_or_spectrum,name=name)
    #else:
    #    Field=field_or_spectrum
    #assert Field.shape==(n_x,n_y)

    figure_base_left = 1   #unit: inch
    figure_base_bottom = 1   #unit: inch
    ax_main_left=figure_base_left
    ax_main_bottom=figure_base_bottom
    ax_main_height=6   #unit: inch
    ax_main_width=ax_main_height*(n_x*d_x)/(n_y*d_y)   #unit: inch
    ax_x_profile_left = figure_base_left   #unit: inch
    ax_x_profile_bottom = figure_base_bottom + ax_main_height   #unit: inch
    ax_x_profile_width = ax_main_width   #unit: inch
    ax_x_profile_height = 2   #unit: inch
    ax_y_profile_left = figure_base_left + ax_main_width   #unit: inch
    ax_y_profile_bottom = figure_base_bottom   #unit: inch
    ax_y_profile_width = 2   #unit: inch
    ax_y_profile_height = ax_main_height   #unit: inch
    ax_cbar_left = ax_y_profile_left + ax_y_profile_width + 1   #unit: inch
    ax_cbar_bottom = figure_base_bottom   #unit: inch
    ax_cbar_width = 0.5   #unit: inch
    ax_cbar_height = ax_main_height   #unit: inch
    figure_width=ax_cbar_left+ax_cbar_width+1   #unit: inch
    figure_height=figure_base_bottom+ax_main_height+ax_x_profile_height+1   #unit: inch
    
    
    vmax=Field_envelope_max/laser_Ec
    vmax=1.2
    fig = plt.figure(figsize=(figure_width,figure_height))
    ax_main = fig.add_axes([ax_main_left/figure_width, ax_main_bottom/figure_height, ax_main_width/figure_width, ax_main_height/figure_height])
    pcm = ax_main.pcolormesh(grid_x_axis/laser_lambda, grid_y_axis/laser_lambda, Field_envelope.T/laser_Ec, cmap='Reds', shading='auto',vmin=0,vmax=vmax)
    ax_main.plot(x_center/laser_lambda,y_center/laser_lambda, 'x', markersize=12, markeredgewidth=2, label='center',color='black')
    ax_main.plot([x_left/laser_lambda, x_right/laser_lambda], [y_center/laser_lambda, y_center/laser_lambda], '-', linewidth=2, alpha=0.7,color='black')
    ax_main.plot(x_left/laser_lambda, y_center/laser_lambda, '|', markersize=10, markeredgewidth=2,color='black')
    ax_main.plot(x_right/laser_lambda, y_center/laser_lambda, '|', markersize=10, markeredgewidth=2,color='black')
    ax_main.plot([x_center/laser_lambda, x_center/laser_lambda], [y_left/laser_lambda, y_right/laser_lambda], '-', linewidth=2, alpha=0.7,color='black')
    ax_main.plot(x_center/laser_lambda, y_left/laser_lambda, '_', markersize=10, markeredgewidth=2,color='black')
    ax_main.plot(x_center/laser_lambda, y_right/laser_lambda, '_', markersize=10, markeredgewidth=2,color='black')
    ax_main.plot(grid_x_axis/laser_lambda,grid_x_axis/laser_lambda/D,color='purple', linestyle='--', alpha=0.7)
    ax_main.plot(grid_x_axis/laser_lambda,-grid_x_axis/laser_lambda/D,color='purple', linestyle='--', alpha=0.7,label='λ0/a')
    ax_main.set_xlim(min(grid_x_axis)/laser_lambda,max(grid_x_axis)/laser_lambda)
    ax_main.set_ylim(min(grid_y_axis)/laser_lambda,max(grid_y_axis)/laser_lambda)
    ax_main.legend()
    rect = Rectangle((x_left/laser_lambda, y_left/laser_lambda), x_right/laser_lambda-x_left/laser_lambda, y_right/laser_lambda-y_left/laser_lambda, fill=False, edgecolor='green', linestyle='--', linewidth=1.5)
    ax_main.add_patch(rect)
    ax_main.set_xlabel('x/λ0')
    ax_main.set_ylabel('y/λ0')
    ax_main.set_aspect('equal')
    #ax_main.legend()
    ax_main.set_title('a=E/Ec=B/Bc')


    ax_x_profile = fig.add_axes([ax_x_profile_left/figure_width, ax_x_profile_bottom/figure_height, ax_x_profile_width/figure_width, ax_x_profile_height/figure_height])
    ax_x_profile.set_xlim(ax_main.get_xlim())
    ax_x_profile.plot(grid_x_axis/laser_lambda, Field_envelope_center_x_profile/laser_Ec, linewidth=2,color='red')
    #ax_x_profile.axvline(x_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
    ax_x_profile.axvline(x_left/laser_lambda, color='g', linestyle='--', alpha=0.7)
    ax_x_profile.axvline(x_right/laser_lambda, color='g', linestyle='--', alpha=0.7, label=f'duration={Field_envelope_center_x_profile_FWHM/laser_period:.2f}·T0')
    ax_x_profile.hlines(y=Field_envelope_center_x_profile_peak[1]/laser_Ec,xmin=x_left/laser_lambda,xmax=x_right/laser_lambda, linestyle='--', alpha=0.7, label='√2/2',color='black')
    ax_x_profile.set_xlabel('x/λ0')
    ax_x_profile.set_ylabel('a=E/Ec=B/Bc')
    ax_x_profile.set_ylim(0,vmax)
    ax_x_profile.legend()
    ax_x_profile.grid(True, alpha=0.3)
    ax_x_profile.set_title('y=%.2f·λ0'%(y_center/laser_lambda))
    ax_x_profile.xaxis.set_ticks_position('top')
    ax_x_profile.xaxis.set_label_position('top')
    


    ax_y_profile = fig.add_axes([ax_y_profile_left/figure_width, ax_y_profile_bottom/figure_height, ax_y_profile_width/figure_width, ax_y_profile_height/figure_height])
    ax_y_profile.set_ylim(ax_main.get_ylim())
    ax_y_profile.plot( Field_envelope_center_y_profile/laser_Ec, grid_y_axis/laser_lambda,'-', linewidth=2,color='red')
    #ax_y_profile.axhline(y_center/laser_lambda, color='k', linestyle='--', alpha=0.7)
    ax_y_profile.axhline(y_left/laser_lambda, color='g', linestyle='--', alpha=0.7)
    ax_y_profile.axhline(y_right/laser_lambda, color='g', linestyle='--', alpha=0.7,label=f'waist={Field_envelope_center_y_profile_peak_waist/laser_lambda:.2f}·λ0')
    ax_y_profile.axhline(x_center/D/laser_lambda, color='purple', linestyle='--', alpha=0.7,label='λ0/a')
    ax_y_profile.axhline(-x_center/D/laser_lambda, color='purple', linestyle='--', alpha=0.7)
    ax_y_profile.vlines(x=Field_envelope_center_y_profile_peak[1]/laser_Ec,ymin=y_left/laser_lambda,ymax=y_right/laser_lambda, linestyle='--', alpha=0.7, label='exp(-1)',color='black')
    ax_y_profile.set_ylabel('y/λ0')
    ax_y_profile.set_xlabel('a=E/Ec=B/Bc')
    ax_y_profile.set_xlim(0,vmax)
    ax_y_profile.legend()
    ax_y_profile.grid(True, alpha=0.3)
    ax_y_profile.set_title('x=%.2f·λ0'%(x_center/laser_lambda))
    ax_y_profile.yaxis.set_ticks_position('right') 
    ax_y_profile.yaxis.set_label_position('right') 


    ax_cbar = fig.add_axes([ax_cbar_left/figure_width, ax_cbar_bottom/figure_height, ax_cbar_width/figure_width, ax_cbar_height/figure_height])
    plt.colorbar(pcm, cax=ax_cbar)
    ax_cbar.set_ylabel('a=E/Ec=B/Bc')
    fig.suptitle(f'Field envelope {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir,f'Field_envelope_width_{name}.png'))
    print(os.path.join(working_dir,f'Field_envelope_width_{name}.png'))
    plt.close(fig)
    plt.clf()
    



    return {
        'Field_envelope_max':Field_envelope_max,
        #'Field_envelope_max_id':Field_envelope_max_id,
        #'Field_envelope_max_at_x':grid_x_axis[Field_envelope_max_id[0]].item(),
        #'Field_envelope_max_at_y':grid_y_axis[Field_envelope_max_id[1]].item(),
        #'Field_envelope_moment_1_id':(Field_envelope_moment_1_x_id,Field_envelope_moment_1_y_id),
        'Field_envelope_center_y_profile_peak_waist':float(Field_envelope_center_y_profile_peak_waist),
        'Field_envelope_center_y_profile_peak_left':float(y_left),
        'Field_envelope_center_y_profile_peak_right':float(y_right),
        'Field_envelope_center_x_profile_FWHM':float(Field_envelope_center_x_profile_FWHM),
        #'Field_envelope_at_moment_1':Field_envelope_at_moment_1,
        #'Field_envelope_center_y_profile':Field_envelope_center_y_profile.tolist(),
    }



#@profile
def get_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    Field_continuation=continue_field_2D(Field,n_continuation_x=n_continuation,n_continuation_y=n_continuation,smooth=False)
    Field_spectrum=fftshift(fft2(Field_continuation))*d_x*d_x   #Unit: V·m for E or V·s for B
    return Field_spectrum
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    print('Max amp:',jnp.max(jnp.abs(Field))/laser_amp)
    print('spectrum peak: %f theoretical peak'%(jnp.max(Field_spectrum_square)/laser_spectrum_peak**2))
    print('Total energy (field) %f theoretical energy'%(square_integral_field_2D(Field=Field,d_x=d_x,d_y=d_x)/laser_energy))
    print('Total energy (spectrum) %f theoretical energy'%(square_integral_field_2D(Field=Field_spectrum,d_x=d_f,d_y=d_f,complex_array=True)/laser_energy))
    #plot_field(Field,name=name)
    plot_2D_spectrum(Field_spectrum_square[freq_center_mask],freq_x_axis[freq_axis_center_mask],freq_y_axis[freq_axis_center_mask],name=name)
    return Field_spectrum

def get_field_from_spectrum(Field_spectrum:jnp.ndarray,name=''):
    assert Field_spectrum.shape==(n_continuation,n_continuation)
    Field=jnp.real(ifft2(ifftshift(Field_spectrum))[grid_center_mask])*n_continuation*d_f*n_continuation*d_f
    return Field

    
def get_x_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    Field_spectrum=get_spectrum(Field=Field,name=name)
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    Field_spectrum_square_on_x=jnp.sum(Field_spectrum_square,axis=1)*d_f
    Field_spectrum_square_on_y=jnp.sum(Field_spectrum_square,axis=0)*d_f
    Field_spectrum_square_centerline=Field_spectrum_square[:,n_continuation//2]
    print(jnp.max(Field_spectrum_square_on_x)/laser_spectrum_on_x_peak)
    print(jnp.max(Field_spectrum_square_on_y)/laser_spectrum_on_y_peak)
    print(jnp.max(Field_spectrum_square_centerline)/laser_spectrum_peak**2)
    print(freq_x_axis[jnp.where(Field_spectrum_square_on_x==jnp.max(Field_spectrum_square_on_x))[0]]*laser_lambda)
    print(jnp.sum(Field_spectrum_square_on_x)*d_f/laser_energy)
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_on_x/laser_spectrum_on_x_peak,label='spectrum_on_x')
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_centerline/laser_spectrum_peak**2,label='spectrum_centerline')
    plt.xlabel(xlabel='kx/k0')
    plt.ylabel(ylabel='I(kx)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_x_%s.png' %(name)))
    plt.clf()
    
    plt.semilogy(freq_x_axis*laser_lambda,Field_spectrum_square_on_y/laser_spectrum_on_y_peak,label='spectrum_on_y')
    plt.xlabel(xlabel='ky/k0')
    plt.ylabel(ylabel='I(ky)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(-highest_harmonic,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_y_%s.png' %(name)))
    plt.clf()
    return Field_spectrum_square_on_x


def get_polar_spectrum(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    Field_spectrum=get_spectrum(Field=Field,name=name)
    Field_spectrum_square=jnp.square(jnp.abs(Field_spectrum))
    Field_spectrum_square_polar=cv2.warpPolar(src=np.asarray(Field_spectrum_square[freq_center_mask].T),dsize=(n_freq_radius,n_freq_angle),center=(n_freq_radius,n_freq_radius),maxRadius=n_freq_radius,flags=cv2.WARP_POLAR_LINEAR)   #Field_spectrum_square_polar.shape=(n_freq_angle,n_freq_radius)
    Field_spectrum_square_on_radius=jnp.sum(Field_spectrum_square_polar*freq_r,axis=0)*d_f_angle/2
    Field_spectrum_square_on_angle=jnp.sum(Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    print(jnp.sum(Field_spectrum_square_on_radius)*d_f_radius*2/laser_energy)
    print(jnp.sum(Field_spectrum_square_on_angle)*d_f_angle/laser_energy)
    print(freq_r_axis[jnp.where(Field_spectrum_square_on_radius==jnp.max(Field_spectrum_square_on_radius))[0]]*laser_lambda)
    print(freq_a_axis[jnp.where(Field_spectrum_square_on_angle==jnp.max(Field_spectrum_square_on_angle))[0]])
    print(jnp.max(Field_spectrum_square_polar)/laser_spectrum_peak**2)
    print(jnp.max(Field_spectrum_square_on_radius)/laser_spectrum_on_x_peak)
    print(jnp.max(Field_spectrum_square_on_angle)/(laser_spectrum_on_y_peak*laser_f0/2))
    pd.DataFrame(data={'I(kr)/I0':Field_spectrum_square_on_radius/laser_spectrum_on_x_peak,'kr/k0':freq_r_axis/laser_f0}).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Field_spectrum_square_on_radius_%s.png' %(name))
    plt.semilogy(freq_r_axis/laser_f0,Field_spectrum_square_on_radius/laser_spectrum_on_x_peak)
    plt.xlabel(xlabel='kr/k0')
    plt.ylabel(ylabel='I(kr)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,20)
    plt.ylim(1e-6,1)
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_radius_%s.png' %(name)))
    plt.clf()
    plt.semilogy(freq_a_axis,Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2))
    plt.xlabel(xlabel='kθ')
    plt.ylabel(ylabel='I(kθ)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,2*C.pi)
    plt.ylim(1e-6,1)
    plt.savefig(os.path.join(working_dir,'Spectrum_square_on_angle_%s.png' %(name)))
    plt.clf()
    return Field_spectrum_square_polar,Field_spectrum_square_on_radius
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.semilogy(freq_a_axis, Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2))
    ax.set_ylim(1e-6,1)
    plt.title('Angular energy distribution %s' %(name))
    plt.savefig(os.path.join(working_dir,'Field_spectrum_square_polar_distribution_%s.png' %(name)))
    plt.clf()
    plt.close(fig)
    
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_r_axis,zoom=zoom_factor)/laser_f0,zoom(freq_a_axis,zoom=zoom_factor),zoom(Field_spectrum_square_polar,zoom=zoom_factor)/laser_spectrum_peak**2,cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(0,5)
    ax.set_ylim(0,2*C.pi)
    ax.set_xlabel('kr(k0)')
    ax.set_ylabel('kθ')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum' %(name))
    plt.savefig(os.path.join(working_dir,'Field_spectrum_square_polar_%s.png' %(name)))
    plt.clf()
    plt.close(fig)

def get_divergent_angle_from_spectrum(Field_Ex:jnp.ndarray,Field_Ey:jnp.ndarray,name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
    """
    assert Field_Ex.shape==(n_field_x,n_field_y)
    assert Field_Ey.shape==(n_field_x,n_field_y)
    Field_Ex_spectrum_square_polar,_=get_polar_spectrum(Field=Field_Ex,name=name)
    Field_Ey_spectrum_square_polar,_=get_polar_spectrum(Field=Field_Ey,name=name)
    Field_spectrum_square_polar=Field_Ex_spectrum_square_polar+Field_Ey_spectrum_square_polar
    Field_spectrum_square_on_angle=jnp.sum(Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    Field_spectrum_square_angle_1_moment=jnp.average(a=freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)],weights=Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    Field_spectrum_square_angle_2_moment=jnp.average(a=jnp.square(freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)]),weights=Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    divergent_angle=jnp.sqrt(Field_spectrum_square_angle_2_moment-jnp.square(Field_spectrum_square_angle_1_moment))*2
    print(Field_spectrum_square_angle_1_moment)
    print('divergent angle: %fθ0' %(divergent_angle/laser_theta0))

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.semilogy(freq_a_axis, Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2),c='r',label='Angular spectrum',linewidth=2)
    #ax.axvline(x=Field_spectrum_square_angle_1_moment,ymin=0,ymax=1,linewidth=1,linestyle='-',c='b')
    ax.axvline(x=Field_spectrum_square_angle_1_moment-divergent_angle,ymin=0,ymax=1,label='divergent angle',linewidth=1,linestyle='--',c='b')
    ax.axvline(x=Field_spectrum_square_angle_1_moment+divergent_angle,ymin=0,ymax=1,linewidth=1,linestyle='--',c='b')
    ax.set_ylim(1e-6,1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    plt.legend()
    plt.title('Angular energy distribution %s' %(name))
    plt.savefig(os.path.join(working_dir,'Field_spectrum_square_polar_distribution_%s.png' %(name)))
    plt.clf()
    plt.close(fig)
    return divergent_angle,Field_spectrum_square_on_angle

def get_energy_flux_on_y(Field:jnp.ndarray,name=''):
    assert Field.shape==(n_field_x,n_field_y)
    perpendicular_energy_distribution=jnp.einsum('ij,ij->j',Field,Field)*d_x
    fig,ax = plt.subplots()
    ax.plot(grid_y_axis/laser_lambda,perpendicular_energy_distribution/laser_centerline_energy)
    ax.set_xlabel('y/λ0')
    #ax.set_ylabel('E/Ec')
    #ax.set_ylim(0,1)
    ax.legend()
    plt.savefig(os.path.join(working_dir,'Field_energy_flux_on_y_%s.png' %(name)))
    plt.clf()
    plt.close(fig)
    

#@profile
def get_evolution_from_spectrum(
    A_plus_spectrum_vector:jnp.ndarray,
    A_minus_spectrum_vector:jnp.ndarray,
    evolution_time=0.0,
    name='',
    ):
    r"""
    Args:
        A_plus_spectrum_vector: F\left\{\mathbit{A}_+\right\}=\frac{1}{2}\left(F\left\{\mathbit{E}\right\}-\hat{\mathbit{k}}\times F\left\{c\mathbit{B}\right\}\right).  Unit: V·m
        A_minus_spectrum_vector: F\left\{\mathbit{A}_-\right\}=\frac{1}{2}\left(F\left\{\mathbit{E}\right\}+\hat{\mathbit{k}}\times F\left\{c\mathbit{B}\right\}\right). Unit: V·m. 
        evolution_time: Unit: s
    """
    assert A_plus_spectrum_vector.shape==(3,n_continuation,n_continuation)
    assert A_minus_spectrum_vector.shape==(3,n_continuation,n_continuation)
    time_phase_shift=2*C.pi*freq_radius*C.speed_of_light*evolution_time   #2πc|f|t. shape=(n_continuation,n_continuation)
    A_plus_spectrum_evolution_vector=jnp.einsum('ijk,jk->ijk',A_plus_spectrum_vector,jnp.exp(-1j*time_phase_shift))
    A_minus_spectrum_evolution_vector=jnp.einsum('ijk,jk->ijk',A_minus_spectrum_vector,jnp.exp(1j*time_phase_shift))
    Electric_Field_spectrum_evolution_vector=(A_plus_spectrum_evolution_vector+A_minus_spectrum_evolution_vector)   #Unit: V·m
    Magnetic_Field_spectrum_evolution_vector=jnp.cross(a=freq_unit_vector,b=(A_plus_spectrum_evolution_vector-A_minus_spectrum_evolution_vector),axisa=0,axisb=0,axisc=0)   #Unit: V·m
    return Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector

#@profile
def get_evolution(
    Electric_Field_Ex=jnp.zeros(shape=(n_field_x,n_field_y)),
    Electric_Field_Ey=jnp.zeros(shape=(n_field_x,n_field_y)),
    Electric_Field_Ez=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_Bx=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_By=jnp.zeros(shape=(n_field_x,n_field_y)),
    Magnetic_Field_Bz=jnp.zeros(shape=(n_field_x,n_field_y)),
    evolution_time_list=[0.0],
    window_shift_velocity=jnp.array((0.0,0.0,0.0)),
    grid_x_axis=grid_x_axis,
    grid_y_axis=grid_y_axis,
    name='',
    ):
    """
    Get the evolution of the EM field via phase shift.
    Args:
        Electric_Field_Ex: Unit: V/m
        Electric_Field_Ey: Unit: V/m
        Electric_Field_Ez: Unit: V/m
        Magnetic_Field_Bx: Unit: T
        Magnetic_Field_By: Unit: T
        Magnetic_Field_Bz: Unit: T
        evolution_time_list: List of the evolution time. Unit: s
        window_shift_velocity: The (3d) velocity of the window following the evolution of the field. (vx,vy,vz=0). Unit: m/s
    """
    assert Electric_Field_Ex.shape==(n_field_x,n_field_y)
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y)
    assert Electric_Field_Ez.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_Bx.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_By.shape==(n_field_x,n_field_y)
    assert Magnetic_Field_Bz.shape==(n_field_x,n_field_y)
    window_shift_velocity=jnp.array(window_shift_velocity)
    evolution_time_list=jnp.array(evolution_time_list).flatten()
    assert evolution_time_list.size>0
    assert window_shift_velocity.shape==(3,)
    Electric_Field_Ex_spectrum=get_spectrum(Electric_Field_Ex,name='Ex_%s' %(name))
    Electric_Field_Ey_spectrum=get_spectrum(Electric_Field_Ey,name='Ey_%s' %(name))
    Electric_Field_Ez_spectrum=get_spectrum(Electric_Field_Ez,name='Ez_%s' %(name))
    Magnetic_Field_Bx_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bx,name='Bx_%s' %(name))   #convert the unit of B to E
    Magnetic_Field_By_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_By,name='By_%s' %(name))
    Magnetic_Field_Bz_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bz,name='Bz_%s' %(name))
    Electric_Field_spectrum_vector=jnp.array((Electric_Field_Ex_spectrum,Electric_Field_Ey_spectrum,Electric_Field_Ez_spectrum))   #shape=(3,n_continuation,n_continuation)
    Magnetic_Field_spectrum_vector=jnp.array((Magnetic_Field_Bx_spectrum,Magnetic_Field_By_spectrum,Magnetic_Field_Bz_spectrum))
    freq_unit_vector_cross_Magnetic_Field_spectrum_vector=jnp.cross(a=freq_unit_vector,b=Magnetic_Field_spectrum_vector,axisa=0,axisb=0,axisc=0)
    A_plus_spectrum_vector=(Electric_Field_spectrum_vector-freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2
    A_minus_spectrum_vector=(Electric_Field_spectrum_vector+freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2
    return_list=[]
    for evolution_time in evolution_time_list:
        print(f'evolution time={evolution_time/laser_period:+05.01f}T0')
        window_grid_x_axis=grid_x_axis+window_shift_velocity[0]*evolution_time
        window_grid_y_axis=grid_y_axis+window_shift_velocity[1]*evolution_time
        window_phase_shift=2*C.pi*jnp.einsum('ijk,i->jk',freq_vector,window_shift_velocity)*evolution_time   #2π(fx,fy,fz)·(vx,vy,vz)t. shape=(n_continuation,n_continuation)
        Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector=get_evolution_from_spectrum(
            A_plus_spectrum_vector=A_plus_spectrum_vector,
            A_minus_spectrum_vector=A_minus_spectrum_vector,
            evolution_time=evolution_time
        )
        Electric_Field_spectrum_evolution_in_window_vector=jnp.einsum('ijk,jk->ijk',Electric_Field_spectrum_evolution_vector,jnp.exp(1j*window_phase_shift))
        #Magnetic_Field_spectrum_evolution_in_window_vector=jnp.einsum('ijk,jk->ijk',Magnetic_Field_spectrum_evolution_vector,jnp.exp(1j*window_phase_shift))
        Electric_Field_Ex_evolution_spectrum,Electric_Field_Ey_evolution_spectrum,Electric_Field_Ez_evolution_spectrum=Electric_Field_spectrum_evolution_in_window_vector
        #Magnetic_Field_Bx_evolution_spectrum,Magnetic_Field_By_evolution_spectrum,Magnetic_Field_Bz_evolution_spectrum=Magnetic_Field_spectrum_evolution_in_window_vector
        #Electric_Field_Ex_evolution=get_field_from_spectrum(Electric_Field_Ex_evolution_spectrum)
        Electric_Field_Ey_evolution=get_field_from_spectrum(Electric_Field_Ey_evolution_spectrum)
        Field_analytic_phase=get_envelope(Electric_Field_Ey_evolution_spectrum,'spectrum')['Field_analytic_phase']
        fig,ax = plt.subplots()
        ax=plot_field(
            Electric_Field_Ey_evolution,
            ax=ax,
            grid_x_axis=window_grid_x_axis,
            grid_y_axis=window_grid_y_axis,
            normalize=laser_Ec,
            vmin=-0.05,vmax=0.05,
            #label='φ/π',
            #cmap='hsv',
            name=f'Ey_{name}_{evolution_time/laser_period:+05.01f}T0'
            )
        ax.plot(window_grid_x_axis/laser_lambda,window_grid_x_axis/laser_lambda/D,color='black', linestyle='--', alpha=0.7)
        ax.plot(window_grid_x_axis/laser_lambda,-window_grid_x_axis/laser_lambda/D,color='black', linestyle='--', alpha=0.7,label='λ0/a')
        ax.set_xlim(28,30)
        ax.set_ylim(5,7)
        #ax.set_title('Phase')
        ax.legend()
        plt.savefig(os.path.join(working_dir,f'Ey_{name}_{evolution_time/laser_period:+05.01f}T0_big.png'))
        print(os.path.join(working_dir,f'Ey_{name}_{evolution_time/laser_period:+05.01f}T0.png'))
        plt.close(fig)
        plt.clf()
        continue
        #write_field_2D(Field_list=[Electric_Field_Ex_evolution,Electric_Field_Ey_evolution,Magnetic_Field_Bz_evolution],x_axis=window_grid_x_axis,y_axis=window_grid_y_axis,name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'],nc_name=os.path.join(working_dir,f'%{name}_time={evolution_time/laser_period:+05.01f}T0.nc'))
        return_list.append(get_envelope_width(field_or_spectrum=Electric_Field_Ey_evolution_spectrum,input_type='spectrum',grid_x_axis=window_grid_x_axis,grid_y_axis=window_grid_y_axis,name=f'{name}_time={evolution_time/laser_period:+05.01f}T0'))
    return return_list

data_dict=read_nc(nc_name=os.path.join(working_dir,'Field.nc'),key_name_list=['Electric_Field_Ey','Magnetic_Field_Bz','x','y'])
Electric_Field_Ey=data_dict['Electric_Field_Ey']
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
grid_x_axis=data_dict['x']
grid_y_axis=data_dict['y']


evolution_time_list=jnp.linspace(0*laser_period,50*laser_period,51,endpoint=True)
evolution_time_list=30*laser_period
return_list=get_evolution(
    Electric_Field_Ey=Electric_Field_Ey,
    Magnetic_Field_Bz=Magnetic_Field_Bz,
    evolution_time_list=evolution_time_list,
    window_shift_velocity=(C.speed_of_light,0,0),
    grid_x_axis=grid_x_axis,
    grid_y_axis=grid_y_axis,
    name='propagate'
    )
print(return_list)
pd.DataFrame(data=return_list).to_hdf(path_or_buf=os.path.join(working_dir,'waist.hdf5'),key=f'{D}')
exit(0)

'incident'
'reflection'
'transmission'