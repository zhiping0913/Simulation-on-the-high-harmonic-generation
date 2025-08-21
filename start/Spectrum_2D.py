import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.ndimage import rotate,zoom
from scipy.integrate import simpson
from scipy.signal import peak_widths, find_peaks,hilbert
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import pandas as pd
import cv2
import xarray as xr
from scipy.special import erf 
from start import read_sdf,read_nc,read_dat,write_field_2D


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thin/2D/rotate_field'

theta_degree=0
theta_rad=np.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_E0=(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 40		# Laser field strength
laser_amp=laser_a0*laser_E0
laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
laser_duration=laser_FWHM/math.sqrt(2*math.log(2)) 
laser_Nc=laser_omega**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_w0_lambda= 3
laser_zR_lambda=np.pi*laser_w0_lambda**2
laser_w0=laser_w0_lambda*laser_lambda
laser_zR=laser_zR_lambda*laser_lambda
laser_theta0=1/(np.pi*laser_w0_lambda)
highest_harmonic=50

def theoretical_laser_energy(amp,z_length,w_0):
    return (C.pi/2)*np.square(amp)*z_length*w_0/2


laser_spectrum_peak=C.pi*laser_amp*(laser_duration*C.speed_of_light)*(laser_w0_lambda*laser_lambda)/2
laser_energy=theoretical_laser_energy(amp=laser_amp,z_length=laser_duration*C.speed_of_light,w_0=laser_w0_lambda*laser_lambda)
laser_spectrum_on_x_peak=math.sqrt(C.pi**3/2)*laser_amp**2*(laser_duration*C.speed_of_light)**2*(laser_w0_lambda*laser_lambda)/4
laser_spectrum_on_y_peak=math.sqrt(C.pi**3/2)*laser_amp**2*(laser_duration*C.speed_of_light)*(laser_w0_lambda*laser_lambda)**2/2

a_lambda=0



cells_per_lambda =250
vacuum_length_x_lambda=8   #lambda
vacuum_length_y_lambda=8   #lambda

continuation_length_lambda=20  #lambda
space_length_lambda=2*(max(vacuum_length_x_lambda,vacuum_length_y_lambda)+continuation_length_lambda)
n_field_x=round(2*vacuum_length_x_lambda*cells_per_lambda)
n_field_y=round(2*vacuum_length_y_lambda*cells_per_lambda)
print(n_field_x,n_field_y)
n_continuation=round(space_length_lambda*cells_per_lambda)
n_continuation_x=n_continuation
n_continuation_y=n_continuation
n_freq_radius=round(n_continuation*(highest_harmonic/cells_per_lambda))
n_freq_angle=n_freq_radius*2


d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation. d_f/laser_f0=1/space_length_lambda
d_f_radius=d_f
d_f_angle=2*np.pi/n_freq_angle

x_min=-vacuum_length_x_lambda*laser_lambda
x_max=vacuum_length_x_lambda*laser_lambda
y_min=-vacuum_length_y_lambda*laser_lambda
y_max=vacuum_length_y_lambda*laser_lambda

grid_x_axis=np.linspace(start=x_min,stop=x_max,num=n_field_x,endpoint=False)+d_x/2
grid_y_axis=np.linspace(start=y_min,stop=y_max,num=n_field_y,endpoint=False)+d_x/2
grid_center=np.array([0,0])
grid_x_continuation_axis=np.linspace(start=0,stop=space_length_lambda*laser_lambda,num=n_continuation,endpoint=False)
grid_y_continuation_axis=grid_x_continuation_axis
grid_x,grid_y=np.meshgrid(grid_x_axis,grid_y_axis, indexing='ij')

freq_x_axis=np.fft.fftshift(np.fft.fftfreq(n=n_continuation,d=d_x))
freq_y_axis=freq_x_axis

freq_r_axis=np.fft.fftfreq(n=n_continuation,d=d_x)[0:n_freq_radius]
freq_a_axis=np.linspace(start=0,stop=2*np.pi,num=n_freq_angle,endpoint=False)

freq_a,freq_r=np.meshgrid(freq_a_axis,freq_r_axis,indexing='ij')
freq_x,freq_y=np.meshgrid(freq_x_axis,freq_y_axis, indexing='ij')   #freq_x.shape=(n_continuation,n_continuation)
freq_radius = np.sqrt(freq_x**2 + freq_y**2)   #shape=(n_continuation,n_continuation)
freq_theta = np.arctan2(freq_y,freq_x)

freq_z=np.zeros(shape=(n_continuation,n_continuation),dtype=np.float64)
freq_vector=np.array((freq_x,freq_y,freq_z))   #shape=(3,n_continuation,n_continuation)
freq_radius_nonzero = freq_radius.copy()
freq_radius_nonzero[freq_radius == 0] = 1e-20   #Avoid divided by zero
freq_unit_vector=freq_vector/freq_radius_nonzero[np.newaxis, :, :]    #shape=(3,n_continuation,n_continuation)


grid_center_mask=np.s_[round((n_continuation_x-n_field_x)/2):round((n_continuation_x+n_field_x)/2),round((n_continuation_y-n_field_y)/2):round((n_continuation_y+n_field_y)/2)]
freq_center_mask=np.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius,n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]
freq_axis_center_mask=np.s_[n_continuation//2-n_freq_radius:n_continuation//2+n_freq_radius]
print(freq_x.shape)
print(freq_x[freq_center_mask].shape)

grid_centerline_axis=grid_x_axis
freq_centerline_axis=freq_x_axis 

#mask_left = grid_x * np.cos(theta_rad) + grid_y * np.sin(theta_rad) < 0
#mask_right = grid_x * np.cos(theta_rad) + grid_y* np.sin(theta_rad) > 0

#mask_left = grid_x-(a_lambda/laser_lambda)*np.square(grid_y)<0
#mask_right=grid_x-(a_lambda/laser_lambda)*np.square(grid_y)>0

filter=3.5

norm = LogNorm(vmin=1e-6, vmax=1.2)
cmap = 'hot'
zoom_factor=0.5

laser_kn=laser_k0


def Gaussian(x,x0,w,A):
    return A*np.exp(-np.square((x-x0)/w))

def theoretical_w_z(z,z_focus,w0):
    global laser_kn
    zR = laser_kn * w0**2 /2
    return w0 * np.sqrt(1 + np.square((z-z_focus)/zR))

def theoretical_Kappa_z(z,z_focus,w0):
    global laser_kn
    zR = laser_kn * w0**2 /2
    Kappa_Z=(z-z_focus)/(np.square(z-z_focus)+np.square(zR))
    return Kappa_Z

def gaussian_beam_profile(x_y,A,z_focus,z_center,w0,z_length):
    x,y=x_y
    w_z=theoretical_w_z(z=x,z_focus=z_focus,w0=w0)
    Electric_Field_envelope=A*np.sqrt(w0/w_z)*np.exp(-np.square(y/w_z))*np.exp(-np.square((x-z_center)/z_length))
    return Electric_Field_envelope

def w_z_residuals(params, x, y, weights):
    return weights * (y - theoretical_w_z(x, *params))

def theoretical_transversal_field_envelope(x,z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    transversal_field_envelope=np.power(1+np.square((z-z_focus)/zR),-1/4)*np.exp(-np.square(x/w_z))
    return transversal_field_envelope

def theoretical_longitudinal_field_envelope(x,z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    w_z=theoretical_w_z(z,z_focus,w0)
    longitudinal_field_envelope=np.power(1+np.square((z-z_focus)/zR),-3/4)*np.exp(-np.square(x/w_z))*x/zR
    return longitudinal_field_envelope

def Gouy_phase(z,z_focus,w0):
    """
        The wave is traveling in +z direction with polarisation in +x direction.
    """
    global laser_kn
    zR = laser_kn * w0**2 /2
    return np.atan((z-z_focus)/zR)

def continue_field(Electric_Field:np.ndarray,n_continuation_x=n_continuation_x,n_continuation_y=n_continuation_y,edge_lambda=2,name=''):
    """
        Extend the field array to a shape (n_continuation_x,n_continuation_y) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_lambda: the length of the smoothing area at the edge. unit: laser_lambda
    """
    n_x,n_y=Electric_Field.shape
    assert n_x<=n_continuation_x
    assert n_y<=n_continuation_y
    grid_center_mask=np.s_[round((n_continuation_x-n_x)/2):round((n_continuation_x+n_x)/2),round((n_continuation_y-n_y)/2):round((n_continuation_y+n_y)/2)]
    x_id,y_id=np.meshgrid(np.arange(n_x),np.arange(n_y),indexing='ij')
    n_edge_x=edge_lambda*cells_per_lambda
    n_edge_y=edge_lambda*cells_per_lambda
    x_left_trans=0.5 * (1 + erf((x_id - n_edge_x) /n_edge_x))
    x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-n_edge_x)) / n_edge_x))
    y_left_trans=0.5 * (1 + erf((y_id - n_edge_y) /n_edge_y))
    y_right_trans= 0.5 * (1 - erf((y_id - (n_y-1-n_edge_y)) / n_edge_y))   #smooth the edge
    Electric_Field_continuation=np.zeros(shape=(n_continuation_x,n_continuation_y))
    Electric_Field_continuation[grid_center_mask]=Electric_Field*x_left_trans*x_right_trans*y_left_trans*y_right_trans
    return Electric_Field_continuation

def plot_field(Electric_Field:np.ndarray,grid_x_axis=grid_x_axis,grid_y_axis=grid_y_axis,name=''):
    assert Electric_Field.ndim==2
    assert grid_x_axis.ndim==1
    assert grid_y_axis.ndim==1
    assert Electric_Field.shape==(grid_x_axis.size,grid_y_axis.size)
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(grid_x_axis,zoom=zoom_factor)/laser_lambda,zoom(grid_y_axis,zoom=zoom_factor)/laser_lambda,zoom(Electric_Field.T,zoom=zoom_factor)/laser_E0,vmin=-laser_a0,vmax=laser_a0,cmap='RdBu')
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel('a=E/E0')
    plt.xlabel('x/λ0')
    plt.ylabel('y/λ0')
    plt.title(name)
    plt.savefig(os.path.join(working_dir,'Electric_Field_%s.png' %(name)))
    plt.close(fig)
    plt.clf()

def plot_2D_spectrum(Electric_Field_spectrum_square:np.ndarray,freq_x_axis=freq_x_axis,freq_y_axis=freq_y_axis,name=''):
    assert Electric_Field_spectrum_square.ndim==2
    assert freq_x_axis.ndim==1
    assert freq_y_axis.ndim==1
    assert Electric_Field_spectrum_square.shape==(freq_x_axis.size,freq_y_axis.size)
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_x_axis,zoom=zoom_factor)/laser_f0,zoom(freq_y_axis,zoom=zoom_factor)/laser_f0,zoom(Electric_Field_spectrum_square.T,zoom=zoom_factor)/laser_spectrum_peak**2,cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('kx/k0')
    ax.set_ylabel('ky/k0')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum I(kx,ky)/I0' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_%s.png' %(name)))
    plt.clf()


def filter_spectrum(Electric_Field_spectrum:np.ndarray,filter_range=(1.5,highest_harmonic),name=''):
    global laser_kn
    laser_kn=laser_k0*np.average(filter_range)
    assert Electric_Field_spectrum.shape==(n_continuation,n_continuation)
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    spectrum_mask=(freq_radius*laser_lambda>filter_range[0])&(freq_radius*laser_lambda<filter_range[1])#&freq_near_axis
    Electric_Field_filter_spectrum=Electric_Field_spectrum*spectrum_mask
    Electric_Field_filter_spectrum_square=np.square(np.abs(Electric_Field_filter_spectrum))
    Electric_Field_filter=np.real(np.fft.ifft2(np.fft.ifftshift(Electric_Field_filter_spectrum))[grid_center_mask])*n_continuation*d_f*n_continuation*d_f
    print('total energy: %f theoretical total energy' %(np.sum(Electric_Field_spectrum_square)*d_f*d_f/laser_energy))
    print('filter energy: %f theoretical total energy' %(np.sum(Electric_Field_filter_spectrum_square)*d_f*d_f/laser_energy))
    return Electric_Field_filter, Electric_Field_filter_spectrum
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(grid_x_axis[0:n_field_x//2],zoom=zoom_factor)/laser_lambda,zoom(grid_y_axis,zoom=zoom_factor)/laser_lambda,zoom(Electric_Field_filter[0:n_field_x//2,:].T,zoom=zoom_factor)/laser_amp,vmin=-1,vmax=1,cmap='RdBu')
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel('Ey(E0)')
    plt.xlabel('x/λ0')
    plt.ylabel('y/λ0')
    plt.savefig(os.path.join(working_dir,'Electric_Field_filter_%s.png' %(name)))
    plt.close(fig)
    plt.clf()
    return Electric_Field_filter, Electric_Field_filter_spectrum
    plt.plot(grid_x_axis/laser_lambda,Electric_Field_filter[n_field_y//2]/laser_amp)
    plt.xlabel('x/λ0')
    plt.ylabel('E_filter (E0)')
    #plt.ylim(-0.3,0.3)
    plt.title('centerline of filtered field')
    plt.savefig(os.path.join(working_dir,'Electric_Field_filter_centerline_%s.png' %(name)))
    plt.clf()
    #return Electric_Field_filter, Electric_Field_filter_spectrum

    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_x_axis[freq_axis_center_mask],zoom=zoom_factor)/laser_f0,zoom(freq_y_axis[freq_axis_center_mask],zoom=zoom_factor)/laser_f0,zoom(Electric_Field_filter_spectrum_square[freq_center_mask].T,zoom=zoom_factor)/laser_spectrum_peak**2,cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('kx/k0')
    ax.set_ylabel('ky/k0')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_filter_spectrum_square_%s.png' %(name)))
    plt.clf()


    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_x_axis[freq_axis_center_mask],zoom=zoom_factor)/laser_f0,zoom(freq_y_axis[freq_axis_center_mask],zoom=zoom_factor)/laser_f0,zoom(Electric_Field_spectrum_square[freq_center_mask].T,zoom=zoom_factor)/laser_spectrum_peak**2,cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_xlabel('kx/k0')
    ax.set_ylabel('ky/k0')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_%s.png' %(name)))
    plt.clf()

    return Electric_Field_filter, Electric_Field_filter_spectrum

def get_envelope(Electric_Field:np.ndarray,name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
    """
    assert Electric_Field.shape==(n_field_x,n_field_y)
    Electric_Field_continuation=continue_field(Electric_Field)
    Electric_Field_analytic=hilbert(Electric_Field_continuation,axis=0)[round((n_continuation-n_field_x)/2):round((n_continuation+n_field_x)/2),round((n_continuation-n_field_y)/2):round((n_continuation+n_field_y)/2)]
    Electric_Field_analytic_phase=np.angle(Electric_Field_analytic)
    Electric_Field_envelope=np.abs(Electric_Field_analytic)
    Electric_Field_envelope_max=np.max(Electric_Field_envelope)
    Electric_Field_envelope_max_id=tuple(np.array(np.where(Electric_Field_envelope==Electric_Field_envelope_max))[:,0])   #Electric_Field_envelope_max_id=(x_id,y_id)
    print(Electric_Field_envelope_max/laser_amp)
    print(Electric_Field_envelope_max_id)
    #Electric_Field_envelope_max_id=(n_field_x//2,n_field_y//2)
    Electric_Field_envelope_max_phase=Electric_Field_analytic_phase[Electric_Field_envelope_max_id]
    print('Phase at the peak: %fπ'%(Electric_Field_envelope_max_phase/np.pi))
    return {
        'Electric_Field_envelope':Electric_Field_envelope,
        'Electric_Field_envelope_max':Electric_Field_envelope_max,
        'Electric_Field_envelope_max_id':Electric_Field_envelope_max_id,
        'Electric_Field_envelope_max_phase':Electric_Field_envelope_max_phase,
    }
    
    Electric_Field_analytic_phase_centerline=Electric_Field_analytic_phase[:,Electric_Field_envelope_max_id[1]]
    Electric_Field_analytic_phase_unwrap=np.unwrap(Electric_Field_analytic_phase,axis=0,period=2*np.pi)
    Electric_Field_analytic_phase_unwrap=Electric_Field_analytic_phase_unwrap-Electric_Field_analytic_phase_unwrap[Electric_Field_envelope_max_id]+Electric_Field_envelope_max_phase   #Phase relative to the peak. Keep the phase at the peak
    Electric_Field_propagate_phase=laser_k0*(grid_x-grid_x[Electric_Field_envelope_max_id])+Electric_Field_envelope_max_phase
    Electric_Field_phase_extra=Electric_Field_analytic_phase_unwrap-Electric_Field_propagate_phase   # Gouy phase
    Electric_Field_phase_gradient=np.gradient(Electric_Field_phase_extra,grid_x_axis,axis=0)
    zR=-0.5/(np.average(Electric_Field_phase_gradient[Electric_Field_envelope_max_id[0]-5:Electric_Field_envelope_max_id[0]+105,Electric_Field_envelope_max_id[1]]))
    W0=np.sqrt(laser_lambda*zR/np.pi)
    print(zR/laser_lambda)
    print(W0/laser_lambda)
    plt.plot(grid_x_axis/laser_lambda,Electric_Field[:,Electric_Field_envelope_max_id[1]]/laser_amp,c='b',label='field',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda,Electric_Field_envelope[:,Electric_Field_envelope_max_id[1]]/laser_amp,linestyle='--',c='g',label='envelope',linewidth=1)
    plt.plot(grid_x_axis/laser_lambda,-Electric_Field_envelope[:,Electric_Field_envelope_max_id[1]]/laser_amp,linestyle='--',c='g',linewidth=1)
    #plt.xlim(grid_x_axis[Electric_Field_envelope_max_id[0]]/laser_lambda-1,grid_x_axis[Electric_Field_envelope_max_id[0]]/laser_lambda+1)
    #plt.ylim(-1,1)
    plt.legend()
    plt.xlabel(xlabel='x/λ0')
    plt.ylabel(ylabel='E/E0')
    plt.title(label='Field at the centerline %s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_centerline_%s.png' %(name)))
    plt.clf()

    
    plt.plot(grid_x_axis/laser_lambda,Electric_Field_phase_extra[:,Electric_Field_envelope_max_id[1]]/np.pi,label='envelope phase',c='b',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda,-0.5*np.atan((grid_x_axis-grid_x_axis[Electric_Field_envelope_max_id[0]])/zR)/np.pi,label='Gouy phase',linestyle='--',c='g',linewidth=1)
    #plt.xlim(grid_x_axis[Electric_Field_envelope_max_id[0]]/laser_lambda-1,grid_x_axis[Electric_Field_envelope_max_id[0]]/laser_lambda+1)
    plt.ylim(-1,1)
    plt.xlabel(xlabel='x_L/λ0')
    plt.ylabel(ylabel='phase/π')
    plt.legend()
    plt.title(label='Phase_%s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_phase_extra_%s.png' %(name)))
    plt.clf()



    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(grid_x_axis/laser_lambda,grid_y_axis/laser_lambda,Electric_Field_envelope.T/laser_amp,cmap='Reds')
    ax.set_aspect('equal')
    plt.colorbar(pcm).ax.set_ylabel('envelope(E0)')
    plt.xlabel('x/λ0')
    plt.ylabel('y/λ0')
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_%s.png' %(name)))
    plt.close(fig)
    plt.clf()


def get_envelope_width(Electric_Field_envelope:np.ndarray,name=''):
    assert Electric_Field_envelope.shape==(n_field_x,n_field_y)
    Electric_Field_envelope_square=np.square(Electric_Field_envelope)
    Electric_Field_envelope_x_1_moment=np.average(a=grid_x,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_2_moment=np.average(a=np.square(grid_x),weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_std=np.sqrt(Electric_Field_envelope_x_2_moment-np.square(Electric_Field_envelope_x_1_moment))
    print('FWHM obtained from envelope_x_std: %ffs'%(Electric_Field_envelope_x_std*2*math.sqrt(2*math.log(2))/C.speed_of_light/C.femto))
    Electric_Field_envelope_y_1_moment=np.average(a=grid_y,axis=1,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_y_2_moment=np.average(a=np.square(grid_y),axis=1,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_y_std=np.sqrt(Electric_Field_envelope_y_2_moment-np.square(Electric_Field_envelope_y_1_moment))
    Electric_Field_envelope_wz=2*Electric_Field_envelope_y_std
    Electric_Field_envelope_wz_min=np.min(Electric_Field_envelope_wz)
    print('min waist from Electric_Field_envelope_y_std: %fλ0' %(Electric_Field_envelope_wz_min/laser_lambda))
    print(np.where(Electric_Field_envelope_wz==Electric_Field_envelope_wz_min))
    Electric_Field_envelope_max=np.max(Electric_Field_envelope)
    Electric_Field_envelope_max_id=tuple(np.array(np.where(Electric_Field_envelope==Electric_Field_envelope_max))[:,0])   #Electric_Field_envelope_max_id=(x_id,y_id)
    Electric_Field_envelope_centercross=Electric_Field_envelope[Electric_Field_envelope_max_id[0],:]
    Electric_Field_envelope_centerline=Electric_Field_envelope[:,Electric_Field_envelope_max_id[1]]
    Electric_Field_envelope_centercross_peak=peak_widths(x=Electric_Field_envelope_centercross,peaks=[Electric_Field_envelope_max_id[1]],rel_height=1-np.exp(-1))
    Electric_Field_envelope_centerline_peak=peak_widths(x=np.square(Electric_Field_envelope_centerline),peaks=[Electric_Field_envelope_max_id[0]],rel_height=0.5)
    Electric_Field_envelope_centercross_peak_width=Electric_Field_envelope_centercross_peak[0].item()*d_x   #unit: m
    Electric_Field_envelope_centerline_peak_width=Electric_Field_envelope_centerline_peak[0].item()*d_x   #unit: m
    Electric_Field_envelope_centercross_peak_waist=Electric_Field_envelope_centercross_peak_width/2
    print('waist at the peak: %fλ0' %(Electric_Field_envelope_centercross_peak_waist/laser_lambda))
    print('centerline FWHM: %ffs' %(Electric_Field_envelope_centerline_peak_width/C.speed_of_light/C.femto))
    
    return {
        'Electric_Field_envelope_wz':Electric_Field_envelope_wz,
        'Electric_Field_envelope_wz_min':Electric_Field_envelope_wz_min,
        'Electric_Field_envelope_centercross_peak_waist':Electric_Field_envelope_centercross_peak_waist,
    }
    

def get_focus(Electric_Field:np.ndarray,filter_range=(1.5,highest_harmonic),name=''):
    global laser_kn
    laser_kn=laser_k0*np.average(filter_range)
    assert Electric_Field.shape==(n_field_x,n_field_y)
    Electric_Field_spectrum=get_spectrum(Electric_Field,name=name)
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    divergent_angle=np.sqrt(np.average(a=np.square(freq_theta[n_continuation//2:,:]),weights=Electric_Field_spectrum_square[n_continuation//2:,:]))*2
    spectrum_center_radius=np.average(a=freq_radius,weights=Electric_Field_spectrum_square)
    divergent_angle_waist=1/(spectrum_center_radius*np.pi*divergent_angle)
    spectrum_width=np.sqrt(np.average(a=np.square(freq_radius),weights=Electric_Field_spectrum_square)-np.square(spectrum_center_radius))
    print('divergent angle (half angle) θ=%f' %(divergent_angle))
    print('average w=λ/(πθ)=%f×λ0' %(divergent_angle_waist/laser_lambda))
    print('spectrum_center_radius %f×f0' %(spectrum_center_radius/laser_f0))
    print('spectrum width f_std=%f(m^-1)' %(spectrum_width))
    print('FWHM duration τ=sqrt(2ln2)/(2*pi*c*f_std)=%ffs' %(math.sqrt(2*math.log(2))/(2*C.pi*spectrum_width*C.speed_of_light)/C.femto))
    #return divergent_angle, divergent_angle_waist/laser_lambda,np.sum(Electric_Field_spectrum_square)*d_f*d_f/laser_energy
    Electric_Field_envelope=get_envelope(Electric_Field,name=name)['Electric_Field_envelope']
    Electric_Field_envelope_square=np.square(Electric_Field_envelope)
    Electric_Field_envelope_x_1_moment=np.average(a=grid_x,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_2_moment=np.average(a=np.square(grid_x),weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_std=np.sqrt(Electric_Field_envelope_x_2_moment-np.square(Electric_Field_envelope_x_1_moment))
    print('FWHM obtained from envelope_x_std: %ffs'%(Electric_Field_envelope_x_std*2*math.sqrt(2*math.log(2))/C.speed_of_light/C.femto))
    Electric_Field_envelope_y_1_moment=np.average(a=grid_y,axis=0,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_y_2_moment=np.average(a=np.square(grid_y),axis=0,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_y_std=np.sqrt(Electric_Field_envelope_y_2_moment-np.square(Electric_Field_envelope_y_1_moment))

    Electric_Field_envelope_max=np.max(Electric_Field_envelope)
    print(Electric_Field_envelope_max/laser_amp)
    Electric_Field_envelope_peak_y_index,Electric_Field_envelope_peak_x_index=np.array(np.where(Electric_Field_envelope==Electric_Field_envelope_max)).T[0]
    Electric_Field_envelope_centerline=Electric_Field_envelope[:,Electric_Field_envelope_peak_y_index]
    Electric_Field_envelope_cross=Electric_Field_envelope[Electric_Field_envelope_peak_x_index,:]
    print(Electric_Field_envelope_peak_y_index,Electric_Field_envelope_peak_x_index)
    print(grid_y_axis[Electric_Field_envelope_peak_y_index],grid_x_axis[Electric_Field_envelope_peak_x_index])
    Electric_Field_envelope_centerline_peak=peak_widths(x=Electric_Field_envelope_centerline,peaks=[Electric_Field_envelope_peak_x_index],rel_height=1-math.exp(-1))
    Electric_Field_envelope_duration_period=Electric_Field_envelope_centerline_peak[0][0]/cells_per_lambda/2   #half width
    Electric_Field_envelope_centerline_peak_left_lambda=-vacuum_length_x_lambda+Electric_Field_envelope_centerline_peak[2][0]/cells_per_lambda
    Electric_Field_envelope_centerline_peak_right_lambda=-vacuum_length_x_lambda+Electric_Field_envelope_centerline_peak[3][0]/cells_per_lambda
    Electric_Field_pulse_x_mask=(grid_x_axis>Electric_Field_envelope_centerline_peak_left_lambda*laser_lambda)&(grid_x_axis<Electric_Field_envelope_centerline_peak_right_lambda*laser_lambda)
    #Electric_Field_pulse_x_mask=Electric_Field_envelope_centerline>-math.exp(-2)*Electric_Field_envelope_max
    Electric_Field_pulse_x=grid_x_axis[Electric_Field_pulse_x_mask]
    print('envelope duration FWHM: %fT=%ffs.' %(Electric_Field_envelope_duration_period*math.sqrt(2*math.log(2)),Electric_Field_envelope_duration_period*math.sqrt(2*math.log(2))*laser_period/C.femto))
    Electric_Field_envelope_square_FWHM=find_peaks(x=np.square(Electric_Field_envelope_centerline),height=np.square(Electric_Field_envelope_max),width=0,rel_height=1/2)[1]['widths']/cells_per_lambda*laser_period
    print(Electric_Field_envelope_square_FWHM/C.femto)

    Electric_Field_envelope_cross_peak=peak_widths(x=Electric_Field_envelope_cross,peaks=[Electric_Field_envelope_peak_y_index],rel_height=1-math.exp(-1))
    Electric_Field_envelope_cross_width_lambda=Electric_Field_envelope_cross_peak[0][0]/cells_per_lambda/2   #half width
    Electric_Field_envelope_cross_peak_left_lambda=-vacuum_length_y_lambda+Electric_Field_envelope_cross_peak[2][0]/cells_per_lambda
    Electric_Field_envelope_cross_peak_right_lambda=-vacuum_length_y_lambda+Electric_Field_envelope_cross_peak[3][0]/cells_per_lambda
    print('envelope cross width: %fλ=%fμm' %(Electric_Field_envelope_cross_width_lambda,Electric_Field_envelope_cross_width_lambda*laser_lambda/C.micron))



    Electric_Field_envelope_fit_y0=np.zeros(shape=(len(Electric_Field_pulse_x)))
    Electric_Field_envelope_fit_waist=np.zeros(shape=(len(Electric_Field_pulse_x)))
    Electric_Field_envelope_peak_y_center_index=np.zeros(shape=(len(Electric_Field_pulse_x),),dtype=np.int64)
    Electric_Field_envelope_peak_y_left_ips=np.zeros(shape=(len(Electric_Field_pulse_x),))
    Electric_Field_envelope_peak_y_right_ips=np.zeros(shape=(len(Electric_Field_pulse_x),))
    Electric_Field_envelope_peak_y_width=np.zeros(shape=(len(Electric_Field_pulse_x),))
    for x_index in range(len(Electric_Field_pulse_x)):
        peaks, properties=find_peaks(x=Electric_Field_envelope[Electric_Field_pulse_x_mask,:][:,x_index],height=np.max(Electric_Field_envelope[Electric_Field_pulse_x_mask,:][:,x_index]),width=0,rel_height=1-math.exp(-1))
        Electric_Field_envelope_peak_y_left_ips[x_index]=properties['left_ips'].item(0)
        Electric_Field_envelope_peak_y_right_ips[x_index]=properties['right_ips'].item(0)
        Electric_Field_envelope_peak_y_center_index[x_index]=peaks[0].item(0)
        Electric_Field_envelope_peak_y_width[x_index]=d_x*properties['widths'].item(0)/2
    Electric_Field_envelope_peak_y_center=-vacuum_length_y_lambda*laser_lambda+d_x*Electric_Field_envelope_peak_y_center_index
    Electric_Field_envelope_peak_y_left=-vacuum_length_y_lambda*laser_lambda+d_x*Electric_Field_envelope_peak_y_left_ips
    Electric_Field_envelope_peak_y_right=-vacuum_length_y_lambda*laser_lambda+d_x*Electric_Field_envelope_peak_y_right_ips
    z0_fit,w0_fit=curve_fit(f=theoretical_w_z,xdata=Electric_Field_pulse_x,ydata=Electric_Field_envelope_peak_y_width,p0=(np.average(Electric_Field_pulse_x),divergent_angle_waist))[0]
    print('z0_fit: %fλ' %(z0_fit/laser_lambda))
    print('w0_fit: %fλ' %(w0_fit/laser_lambda))
    
    Electric_Field_envelope_waist_fit=theoretical_w_z(z=grid_x_axis,z_focus=z0_fit,w0=w0_fit)
    plt.plot(grid_x_axis/laser_lambda,np.square(Electric_Field_envelope_centerline)/laser_amp**2,c='b')
    plt.xlabel('x/λ0')
    plt.ylabel('I (I0)')
    plt.xlim(-vacuum_length_x_lambda,0)
    plt.title('Electric field centerline intensity')
    plt.savefig(os.path.join(working_dir,'Electric_Field_intensity_centerline_%s.png' %(name)))
    plt.clf()
    return 0
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(grid_x_axis,zoom=zoom_factor)/laser_lambda,zoom(grid_y_axis,zoom=zoom_factor)/laser_lambda,zoom(Electric_Field_envelope.T,zoom=zoom_factor)/laser_amp,cmap='Reds',vmax=1.2,vmin=0)
    ax.set_aspect('equal')
    #ax.plot(grid_x_axis/laser_lambda,Electric_Field_envelope_y_1_moment/laser_lambda,label='Electric_Field_envelope_y_1_moment')
    #ax.plot(Electric_Field_pulse_x/laser_lambda,Electric_Field_envelope_fit_y0/laser_lambda,label='Electric_Field_envelope_fit_y0')
    ax.plot(grid_x_axis/laser_lambda,(Electric_Field_envelope_y_1_moment+Electric_Field_envelope_y_std*2)/laser_lambda,label='envelope_std*2')
    ax.plot(Electric_Field_pulse_x/laser_lambda,(Electric_Field_envelope_fit_y0+Electric_Field_envelope_fit_waist)/laser_lambda,label='envelope_fit_waist')
    ax.plot(grid_x_axis/laser_lambda,Electric_Field_envelope_waist_fit/laser_lambda,c='b',label='envelope_waist_fit')
    ax.plot(grid_x_axis/laser_lambda,-Electric_Field_envelope_waist_fit/laser_lambda,c='b')
    ax.vlines(x=z0_fit/laser_lambda,ymin=-w0_fit/laser_lambda,ymax=w0_fit/laser_lambda,colors='black')
    #ax.plot(Electric_Field_pulse_x/laser_lambda,Electric_Field_envelope_peak_y_center/laser_lambda,label='Electric_Field_envelope_peak_y_center')
    #ax.plot(Electric_Field_pulse_x/laser_lambda,Electric_Field_envelope_peak_y_left/laser_lambda,label='Electric_Field_envelope_peak_y_left')
    ax.plot(Electric_Field_pulse_x/laser_lambda,Electric_Field_envelope_peak_y_right/laser_lambda,label='envelope_peak_y_right')
    plt.legend()
    plt.colorbar(pcm).ax.set_ylabel('Electric_Field_envelope(E0)')
    ax.set_xlabel('x/λ0')
    ax.set_ylabel('y/λ0')
    ax.set_xlim(-vacuum_length_x_lambda,0)
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_%s.png' %(name)))
    plt.close(fig)
    plt.clf()
    plt.plot(grid_x_axis/laser_lambda,Electric_Field_envelope_centerline/laser_amp,c='b')
    plt.hlines(y=Electric_Field_envelope_centerline_peak[1][0]/laser_amp,xmin=Electric_Field_envelope_centerline_peak_left_lambda,xmax=Electric_Field_envelope_centerline_peak_right_lambda,linestyles='dashed',colors='g')
    plt.xlabel('x/λ0')
    plt.ylabel('E (E0)')
    plt.title('Electric Field Ey centerline envelope')
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_centerline_%s.png' %(name)))
    plt.clf()
    plt.plot(grid_y_axis/laser_lambda,Electric_Field_envelope_cross/laser_amp,c='b')
    plt.hlines(y=Electric_Field_envelope_cross_peak[1][0]/laser_amp,xmin=Electric_Field_envelope_cross_peak_left_lambda,xmax=Electric_Field_envelope_cross_peak_right_lambda,linestyles='dashed',colors='g')
    plt.xlabel('y/λ0')
    plt.ylabel('E (E0)')
    plt.title('Electric Field Ey cross envelope')
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_cross_%s.png' %(name)))
    plt.clf()










def get_spectrum(Electric_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,n_field_y)
    Electric_Field_continuation=continue_field(Electric_Field)
    Electric_Field_spectrum=np.fft.fftshift(np.fft.fft2(Electric_Field_continuation))*d_x*d_x   #Unit: V·m
    return Electric_Field_spectrum
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    
    print('Max amp:',np.max(np.abs(Electric_Field))/laser_amp)
    print('spectrum peak: %f theoretical peak'%(np.max(Electric_Field_spectrum_square)/laser_spectrum_peak**2))

    print(np.sum(np.square(np.abs(Electric_Field)))*d_x*d_x/laser_energy)
    print(np.sum(Electric_Field_spectrum_square)*d_f*d_f/laser_energy)
    return Electric_Field_spectrum
    
    

    plot_field(Electric_Field,name=name)
    plot_2D_spectrum(Electric_Field_spectrum_square[freq_center_mask],freq_x_axis[freq_axis_center_mask],freq_y_axis[freq_axis_center_mask],name=name)
    return Electric_Field_spectrum


def get_x_spectrum(Electric_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,n_field_y)
    Electric_Field_spectrum=get_spectrum(Electric_Field=Electric_Field,name=name)
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    Electric_Field_spectrum_square_on_x=np.sum(Electric_Field_spectrum_square,axis=1)*d_f
    Electric_Field_spectrum_square_on_y=np.sum(Electric_Field_spectrum_square,axis=0)*d_f
    Electric_Field_spectrum_square_centerline=Electric_Field_spectrum_square[:,n_continuation//2]
    print(np.max(Electric_Field_spectrum_square_on_x)/laser_spectrum_on_x_peak)
    print(np.max(Electric_Field_spectrum_square_on_y)/laser_spectrum_on_y_peak)
    print(np.max(Electric_Field_spectrum_square_centerline)/laser_spectrum_peak**2)
    print(freq_x_axis[np.where(Electric_Field_spectrum_square_on_x==np.max(Electric_Field_spectrum_square_on_x))[0]]*laser_lambda)
    print(np.sum(Electric_Field_spectrum_square_on_x)*d_f/laser_energy)
    plt.semilogy(freq_x_axis*laser_lambda,Electric_Field_spectrum_square_on_x/laser_spectrum_on_x_peak,label='spectrum_on_x')
    plt.semilogy(freq_x_axis*laser_lambda,Electric_Field_spectrum_square_centerline/laser_spectrum_peak**2,label='spectrum_centerline')
    plt.xlabel(xlabel='kx/k0')
    plt.ylabel(ylabel='I(kx)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_on_x_%s.png' %(name)))
    plt.clf()

    plt.semilogy(freq_x_axis*laser_lambda,Electric_Field_spectrum_square_on_y/laser_spectrum_on_y_peak,label='spectrum_on_y')
    plt.xlabel(xlabel='ky/k0')
    plt.ylabel(ylabel='I(ky)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(-highest_harmonic,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_on_y_%s.png' %(name)))
    plt.clf()


def get_polar_spectrum(Electric_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,n_field_y)
    Electric_Field_spectrum=get_spectrum(Electric_Field=Electric_Field,name=name)
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    Electric_Field_spectrum_square_polar=cv2.warpPolar(Electric_Field_spectrum_square[freq_center_mask].T,dsize=(n_freq_radius,n_freq_angle),center=(n_freq_radius,n_freq_radius),maxRadius=n_freq_radius,flags=cv2.WARP_POLAR_LINEAR)   #Electric_Field_spectrum_square_polar.shape=(n_freq_angle,n_freq_radius)
    Electric_Field_spectrum_square_on_radius=np.sum(Electric_Field_spectrum_square_polar*freq_r,axis=0)*d_f_angle/2
    Electric_Field_spectrum_square_on_angle=np.sum(Electric_Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    print(np.sum(Electric_Field_spectrum_square_on_radius)*d_f_radius*2/laser_energy)
    print(np.sum(Electric_Field_spectrum_square_on_angle)*d_f_angle/laser_energy)
    print(freq_r_axis[np.where(Electric_Field_spectrum_square_on_radius==np.max(Electric_Field_spectrum_square_on_radius))[0]]*laser_lambda)
    print(freq_a_axis[np.where(Electric_Field_spectrum_square_on_angle==np.max(Electric_Field_spectrum_square_on_angle))[0]])
    print(np.max(Electric_Field_spectrum_square_polar)/laser_spectrum_peak**2)
    print(np.max(Electric_Field_spectrum_square_on_radius)/laser_spectrum_on_x_peak)
    print(np.max(Electric_Field_spectrum_square_on_angle)/(laser_spectrum_on_y_peak*laser_f0/2))
    pd.DataFrame(data=Electric_Field_spectrum_square_on_radius/laser_spectrum_on_x_peak,index=freq_r_axis/laser_f0).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Electric_Field_spectrum_square_on_radius_%s.png' %(name))
    plt.semilogy(freq_r_axis/laser_f0,Electric_Field_spectrum_square_on_radius/laser_spectrum_on_x_peak)
    plt.xlabel(xlabel='kr/k0')
    plt.ylabel(ylabel='I(kr)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,highest_harmonic)
    plt.ylim(1e-6,1)
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_on_radius_%s.png' %(name)))
    plt.clf()
    plt.semilogy(freq_a_axis,Electric_Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2))
    plt.xlabel(xlabel='kθ')
    plt.ylabel(ylabel='I(kθ)/I0')
    plt.title('%s spectrum' %(name))
    plt.xlim(0,2*np.pi)
    plt.ylim(1e-6,1)
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_on_angle_%s.png' %(name)))
    plt.clf()
    return Electric_Field_spectrum_square_polar,Electric_Field_spectrum_square_on_radius
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.semilogy(freq_a_axis, Electric_Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2))
    ax.set_ylim(1e-6,1)
    plt.title('Angular energy distribution %s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_polar_distribution_%s.png' %(name)))
    plt.clf()
    plt.close(fig)
    
    fig,ax = plt.subplots()
    pcm=ax.pcolormesh(zoom(freq_r_axis,zoom=zoom_factor)/laser_f0,zoom(freq_a_axis,zoom=zoom_factor),zoom(Electric_Field_spectrum_square_polar,zoom=zoom_factor)/laser_spectrum_peak**2,cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlim(0,5)
    ax.set_ylim(0,2*np.pi)
    ax.set_xlabel('kr(k0)')
    ax.set_ylabel('kθ')
    plt.colorbar(pcm).ax.set_ylabel('%s spectrum' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_polar_%s.png' %(name)))
    plt.clf()
    plt.close(fig)

def get_divergent_angle_from_spectrum(Electric_Field_Ex:np.ndarray,Electric_Field_Ey:np.ndarray,name=''):
    """
        Assume the wave is in x direction. If not, please rotate the field first.
    """
    assert Electric_Field_Ex.shape==(n_field_x,n_field_y)
    assert Electric_Field_Ey.shape==(n_field_x,n_field_y)
    Electric_Field_Ex_spectrum_square_polar,_=get_polar_spectrum(Electric_Field=Electric_Field_Ex,name=name)
    Electric_Field_Ey_spectrum_square_polar,_=get_polar_spectrum(Electric_Field=Electric_Field_Ey,name=name)
    Electric_Field_spectrum_square_polar=Electric_Field_Ex_spectrum_square_polar+Electric_Field_Ey_spectrum_square_polar
    Electric_Field_spectrum_square_on_angle=np.sum(Electric_Field_spectrum_square_polar*freq_r,axis=1)*d_f_radius
    Electric_Field_spectrum_square_angle_1_moment=np.average(a=freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)],weights=Electric_Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    Electric_Field_spectrum_square_angle_2_moment=np.average(a=np.square(freq_a_axis[round(n_freq_angle/4):round(n_freq_angle*3/4)]),weights=Electric_Field_spectrum_square_on_angle[round(n_freq_angle/4):round(n_freq_angle*3/4)])
    divergent_angle=np.sqrt(Electric_Field_spectrum_square_angle_2_moment-np.square(Electric_Field_spectrum_square_angle_1_moment))*2
    print(Electric_Field_spectrum_square_angle_1_moment)
    print('divergent angle: %fθ0' %(divergent_angle/laser_theta0))

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.semilogy(freq_a_axis, Electric_Field_spectrum_square_on_angle/(laser_spectrum_on_y_peak*laser_f0/2),c='r',label='Angular spectrum',linewidth=2)
    #ax.axvline(x=Electric_Field_spectrum_square_angle_1_moment,ymin=0,ymax=1,linewidth=1,linestyle='-',c='b')
    ax.axvline(x=Electric_Field_spectrum_square_angle_1_moment-divergent_angle,ymin=0,ymax=1,label='divergent angle',linewidth=1,linestyle='--',c='b')
    ax.axvline(x=Electric_Field_spectrum_square_angle_1_moment+divergent_angle,ymin=0,ymax=1,linewidth=1,linestyle='--',c='b')
    ax.set_ylim(1e-6,1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    plt.legend()
    plt.title('Angular energy distribution %s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_polar_distribution_%s.png' %(name)))
    plt.clf()
    plt.close(fig)
    return divergent_angle,Electric_Field_spectrum_square_on_angle

def get_evolution_from_spectrum(
    A_plus_spectrum_vector:np.ndarray,
    A_minus_spectrum_vector:np.ndarray,
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
    A_plus_spectrum_evolution_vector=np.einsum('ijk,jk->ijk',A_plus_spectrum_vector,np.exp(-1j*time_phase_shift))
    A_minus_spectrum_evolution_vector=np.einsum('ijk,jk->ijk',A_minus_spectrum_vector,np.exp(1j*time_phase_shift))
    Electric_Field_spectrum_evolution_vector=(A_plus_spectrum_evolution_vector+A_minus_spectrum_evolution_vector)
    Magnetic_Field_spectrum_evolution_vector=np.cross(a=freq_unit_vector,b=(A_plus_spectrum_evolution_vector-A_minus_spectrum_evolution_vector),axisa=0,axisb=0,axisc=0)
    return Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector

def get_evolution(
    Electric_Field_Ex=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    Electric_Field_Ey=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    Electric_Field_Ez=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    Magnetic_Field_Bx=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    Magnetic_Field_By=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    Magnetic_Field_Bz=np.zeros(shape=(n_field_x,n_field_y),dtype=np.float64),
    evolution_time_list=[0.0],
    window_shift_velocity=np.array((0.0,0.0,0.0)),
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
    window_shift_velocity=np.array(window_shift_velocity)
    assert window_shift_velocity.shape==(3,)
    Electric_Field_Ex_spectrum=get_spectrum(Electric_Field_Ex,name='Electric_Field_Ex')
    Electric_Field_Ey_spectrum=get_spectrum(Electric_Field_Ey,name='Electric_Field_Ey')
    Electric_Field_Ez_spectrum=get_spectrum(Electric_Field_Ez,name='Electric_Field_Ez')
    Magnetic_Field_Bx_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bx,name='Magnetic_Field_Bx')   #convert the unit of B to E
    Magnetic_Field_By_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_By,name='Magnetic_Field_By')
    Magnetic_Field_Bz_spectrum=get_spectrum(C.speed_of_light*Magnetic_Field_Bz,name='Magnetic_Field_Bz')
    Electric_Field_spectrum_vector=np.array((Electric_Field_Ex_spectrum,Electric_Field_Ey_spectrum,Electric_Field_Ez_spectrum))   #shape=(3,n_continuation,n_continuation)
    Magnetic_Field_spectrum_vector=np.array((Magnetic_Field_Bx_spectrum,Magnetic_Field_By_spectrum,Magnetic_Field_Bz_spectrum))
    freq_unit_vector_cross_Magnetic_Field_spectrum_vector=np.cross(a=freq_unit_vector,b=Magnetic_Field_spectrum_vector,axisa=0,axisb=0,axisc=0)
    A_plus_spectrum_vector=(Electric_Field_spectrum_vector-freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2
    A_minus_spectrum_vector=(Electric_Field_spectrum_vector+freq_unit_vector_cross_Magnetic_Field_spectrum_vector)/2
    Electric_Field_Ex_evolution_list=[]
    Electric_Field_Ey_evolution_list=[]
    for evolution_time in evolution_time_list:
        print('evolution time=%+05.01fT'  %(evolution_time/laser_period))
        window_phase_shift=2*C.pi*np.einsum('ijk,i->jk',freq_vector,window_shift_velocity)*evolution_time   #2π(fx,fy,fz)·(vx,vy,vz)t. shape=(n_continuation,n_continuation)
        Electric_Field_spectrum_evolution_vector,Magnetic_Field_spectrum_evolution_vector=get_evolution_from_spectrum(
            A_plus_spectrum_vector=A_plus_spectrum_vector,
            A_minus_spectrum_vector=A_minus_spectrum_vector,
            evolution_time=evolution_time
        )
        Electric_Field_spectrum_evolution_vector=np.einsum('ijk,jk->ijk',Electric_Field_spectrum_evolution_vector,np.exp(1j*window_phase_shift))
        Magnetic_Field_spectrum_evolution_vector=np.einsum('ijk,jk->ijk',Magnetic_Field_spectrum_evolution_vector,np.exp(1j*window_phase_shift))
        Electric_Field_Ex_evolution_spectrum,Electric_Field_Ey_evolution_spectrum,Electric_Field_Ez_evolution_spectrum=Electric_Field_spectrum_evolution_vector
        #Magnetic_Field_Bx_evolution_spectrum,Magnetic_Field_By_evolution_spectrum,Magnetic_Field_Bz_evolution_spectrum=Magnetic_Field_spectrum_evolution_vector
        Electric_Field_Ex_evolution=np.real(np.fft.ifft2(np.fft.ifftshift(Electric_Field_Ex_evolution_spectrum))[grid_center_mask])*n_continuation*d_f*n_continuation*d_f
        Electric_Field_Ey_evolution=np.real(np.fft.ifft2(np.fft.ifftshift(Electric_Field_Ey_evolution_spectrum))[grid_center_mask])*n_continuation*d_f*n_continuation*d_f
        window_grid_x_axis=grid_x_axis+window_shift_velocity[0]*evolution_time
        window_grid_y_axis=grid_y_axis+window_shift_velocity[1]*evolution_time
        plot_field(Electric_Field_Ey_evolution,grid_x_axis=window_grid_x_axis,grid_y_axis=window_grid_y_axis,name='Ey_%s_time=%+05.01fT' %(name,evolution_time/laser_period))
        Electric_Field_Ex_evolution_list.append(Electric_Field_Ex_evolution)
        Electric_Field_Ey_evolution_list.append(Electric_Field_Ey_evolution)
    return Electric_Field_Ex_evolution_list,Electric_Field_Ey_evolution_list


data_dict=read_nc(nc_name=os.path.join(working_dir,'Field_0023_reflection_250cpl.nc'),key_name_list=['Electric_Field_Ex','Electric_Field_Ey','Magnetic_Field_Bz'])
evolution_times=np.linspace(35*laser_period,75*laser_period,num=41,endpoint=True)
envelope_width_list=[]
Electric_Field_Ex=data_dict['Electric_Field_Ex']
Electric_Field_Ey=data_dict['Electric_Field_Ey']
Magnetic_Field_Bz=data_dict['Magnetic_Field_Bz']
Electric_Field_Ex_evolution_list,Electric_Field_Ey_evolution_list=get_evolution(
    Electric_Field_Ex=Electric_Field_Ex,
    Electric_Field_Ey=Electric_Field_Ey,
    Magnetic_Field_Bz=Magnetic_Field_Bz,
    evolution_time_list=evolution_times,window_shift_velocity=(C.speed_of_light,0,0),name='Reflection')
exit(0)








'incident'
'reflection'
'transmission'