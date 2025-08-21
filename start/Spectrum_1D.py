import sdf_helper
import pprint
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as C
import os
import math
from scipy.signal import hilbert, find_peaks, peak_widths
from numpy.fft import fft, fftshift,ifft,ifftshift
from scipy.optimize import curve_fit
import h5py
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.special import erf 
import xarray as xr
from start import read_sdf, read_nc
theta_degree=45
theta_rad=np.radians(theta_degree)

laser_lambda = 0.8*C.micron		# Laser wavelength
laser_f0=1/laser_lambda
laser_k0=2*C.pi*laser_f0
laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
laser_period=laser_lambda/C.speed_of_light
laser_a0 = 3*math.sqrt(2)		# Laser field strength
laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
laser_FWHM=25*C.femto   #The full width at half maximum of the intensity.
laser_duration=laser_FWHM/math.sqrt(2*math.log(2)) 
laser_Nc=laser_omega**2*C.m_e*C.epsilon_0/C.elementary_charge**2
laser_S0=C.epsilon_0*C.speed_of_light*laser_amp**2/2   #average Poynting vector


cells_per_lambda =4000
vacuum_length_x_lambda=100   #lambda
continuation_length_lambda=2000   #lambda
space_length_lambda=vacuum_length_x_lambda+2*continuation_length_lambda   #lambda
n_field_x=round(vacuum_length_x_lambda*cells_per_lambda)
n_continuation_x=round(space_length_lambda*cells_per_lambda)



d_x=laser_lambda/cells_per_lambda   #unit: m
d_f=1/(space_length_lambda*laser_lambda)   #unit: 1/m, d_x*d_f=1/n_continuation_x



laser_lambda_M=laser_lambda/math.cos(theta_rad)
laser_duration_M=laser_duration/math.cos(theta_rad)
laser_f0_M=laser_f0*math.cos(theta_rad)
laser_k0_M=laser_k0*math.cos(theta_rad)
laser_amp_M=laser_amp*math.cos(theta_rad)
laser_S0_M=laser_S0*math.cos(theta_rad)**2
vacuum_length_x_lambda_M=vacuum_length_x_lambda*math.cos(theta_rad)
space_length_lambda_M=space_length_lambda*math.cos(theta_rad)   #laser_f0_M/d_f
laser_spectrum_peak_M=laser_amp_M*(math.sqrt(C.pi)/2)*(laser_duration_M*C.speed_of_light)*(1-math.exp(-laser_k0_M**2*(laser_duration_M*C.speed_of_light)**2))
laser_energy_M=laser_amp_M**2*math.sqrt(C.pi/2)*(laser_duration_M*C.speed_of_light/2)*(1-math.exp(-laser_k0_M**2*(laser_duration_M*C.speed_of_light)**2/2))

highest_harmonic=1000


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thick/1D'

grid_x_axis=np.linspace(start=0,stop=vacuum_length_x_lambda*laser_lambda,num=n_field_x)+d_x/2   #x_M. unit:m. left or right
grid_x_L_axis=grid_x_axis*math.cos(theta_rad)
grid_x=np.meshgrid(grid_x_axis,indexing='ij')[0]
freq_x_axis=np.fft.fftshift(np.fft.fftfreq(n=n_continuation_x,d=d_x))   #unit: 1/m
freq_x=np.meshgrid(freq_x_axis,indexing='ij')[0]
k_x=2*C.pi*freq_x
freq_radius=np.abs(freq_x)
freq_mask=(freq_x/laser_f0_M>1)&(freq_x/laser_f0_M<highest_harmonic)

grid_center_mask=np.s_[round((n_continuation_x-n_field_x)/2):round((n_continuation_x+n_field_x)/2)]


filter=0
#freq_x_axis[round(n_continuation_x//2+space_length_lambda_M)]≈laser_f0_M




def linear(x, m, c):
    return m * x + c



laser_kn=laser_k0

def reverse_field(Electric_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_reverse=np.flip(Electric_Field)
    return Electric_Field_reverse



def continuation_field(Electric_Field:np.ndarray,n_continuation_x=n_continuation_x,edge_lambda=2,name=''):
    """
        Extend the field array to a shape (n_continuation_x,) for further analysis. The edge of the field array is reduced to 0 to avoid the noice at the edge.
        edge_lambda: the length of the smoothing area at the edge. unit: laser_lambda
    """
    n_x,=Electric_Field.shape
    assert n_x<=n_continuation_x
    grid_center_mask=np.s_[round((n_continuation_x-n_x)/2):round((n_continuation_x+n_x)/2)]
    x_id=np.arange(n_x)
    n_edge_x=edge_lambda*cells_per_lambda
    x_left_trans=0.5 * (1 + erf((x_id - n_edge_x) /n_edge_x))
    x_right_trans= 0.5 * (1 - erf((x_id - (n_x-1-n_edge_x)) / n_edge_x))   #smooth the edge
    Electric_Field_continuation=np.zeros(shape=(n_continuation_x))
    Electric_Field_continuation[grid_center_mask]=Electric_Field*x_left_trans*x_right_trans
    return Electric_Field_continuation
    

def get_polarisation(Electric_Field_Ey:np.ndarray,Electric_Field_Ez:np.ndarray,direction=1,name=''):
    """
        direction: direction>0 means the field travels in +x direction. direction<0 means the field travels in -x direction
    """
    assert Electric_Field_Ey.shape==(n_field_x,)
    assert Electric_Field_Ez.shape==(n_field_x,)
    assert direction!=0
    Electric_Field_Ey_envelope=get_envelope(Electric_Field=Electric_Field_Ey,name=name)['Electric_Field_envelope']
    Electric_Field_Ez_envelope=get_envelope(Electric_Field=Electric_Field_Ez,name=name)['Electric_Field_envelope']
    Electric_Field_envelope_square=np.square(Electric_Field_Ey_envelope)+np.square(Electric_Field_Ez_envelope)
    Electric_Field_envelope_square_max_id=np.argmax(Electric_Field_envelope_square)
    polarisation_field=np.sign(direction)*np.atan2(Electric_Field_Ez,Electric_Field_Ey)
    Electric_Field_Ey_energy=np.sum(np.square(Electric_Field_Ey))*d_x
    Electric_Field_Ez_energy=np.sum(np.square(Electric_Field_Ez))*d_x
    print('Energy in y polarisation: %f theoretical total energy' %(Electric_Field_Ey_energy/laser_energy_M))
    print('Energy in z polarisation: %f theoretical total energy' %(Electric_Field_Ez_energy/laser_energy_M))
    print('arctan(√(Energy_z/Energy_y))=%f×π' %(np.atan2(np.sqrt(Electric_Field_Ez_energy),np.sqrt(Electric_Field_Ey_energy))/C.pi))
    print('polarisation at the center of the pulse: %f×π' %(polarisation_field[Electric_Field_envelope_square_max_id]/C.pi))
    plt.plot(grid_x_axis/laser_lambda_M,polarisation_field/C.pi)
    plt.xlim(grid_x_axis[Electric_Field_envelope_square_max_id]/laser_lambda_M-1,grid_x_axis[Electric_Field_envelope_square_max_id]/laser_lambda_M+1)
    plt.xlabel('x_L/λ0')
    plt.ylabel('polarisation/π')
    plt.title('polarisation')
    plt.savefig(os.path.join(working_dir,'polarisation_%s.png' %(name)))
    plt.clf()
    

def filter_field(Electric_Field:np.ndarray,filter_range=(1.5,highest_harmonic),name=''):
    global laser_kn
    laser_kn=laser_k0*np.average(filter_range)
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_spectrum,Electric_Field_spectrum_square=get_spectrum(Electric_Field=Electric_Field,name=name)
    frequency_mask=(freq_radius/laser_f0_M>filter_range[0])&(freq_radius/laser_f0_M<filter_range[1])
    Electric_Field_filter_spectrum=Electric_Field_spectrum*frequency_mask
    Electric_Field_filter_spectrum_square=np.square(np.abs(Electric_Field_filter_spectrum))
    Electric_Field_filter_continuation=np.real(ifftshift(np.fft.ifft(np.fft.ifftshift(Electric_Field_filter_spectrum))))*n_continuation_x*d_f
    Electric_Field_filter=Electric_Field_filter_continuation[grid_center_mask]
    print('total energy: %f theoretical total energy' %(np.sum(Electric_Field_spectrum_square)*d_f/laser_energy_M))
    print('filter energy: %f theoretical total energy' %(np.sum(Electric_Field_filter_spectrum_square)*d_f/laser_energy_M))
    spectrum_center_radius=np.average(a=freq_radius,weights=Electric_Field_filter_spectrum_square)
    spectrum_width=np.sqrt(np.average(a=np.square(freq_radius),weights=Electric_Field_filter_spectrum_square)-np.square(spectrum_center_radius))
    print('spectrum_center_radius %f×f0' %(spectrum_center_radius/laser_f0_M))
    print('spectrum width f_std=%f(m^-1)' %(spectrum_width))
    #print('FWHM duration τ=sqrt(2ln2)/(2*pi*c*f_std)=%ffs' %(math.sqrt(2*math.log(2))/(2*C.pi*spectrum_width*C.speed_of_light)/C.femto))
    print('spectrum peak: %f theoretical spectrum peak' %(np.sqrt(np.max(Electric_Field_filter_spectrum_square))/laser_spectrum_peak_M))
    Electric_Field_filter_envelope_dict=get_envelope(Electric_Field_filter,name=name)
    Electric_Field_filter_envelope=Electric_Field_filter_envelope_dict['Electric_Field_envelope']
    Electric_Field_filter_envelope_max=Electric_Field_filter_envelope_dict['Electric_Field_envelope_max']
    Electric_Field_filter_envelope_max_id=Electric_Field_filter_envelope_dict['Electric_Field_envelope_max_id']
    Electric_Field_filter_envelope_peak_width=Electric_Field_filter_envelope_dict['Electric_Field_envelope_peak_width']

    figure, ax1 = plt.subplots()
    plt.xlabel(xlabel='x_L/λ0')
    plt.title(label='Field')
    line1,=ax1.plot(grid_x_axis/laser_lambda_M,Electric_Field/laser_amp_M, color='red',label='total')
    ax1.set_ylabel('E_total(E0)', color='red')
    ax1.set_ylim(-1.1,1.1)
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx() 
    line2,=ax2.plot(grid_x_axis/laser_lambda_M,Electric_Field_filter/laser_amp_M, color='blue',label='filter field')
    line3,=ax2.plot(grid_x_axis/laser_lambda_M,Electric_Field_filter_envelope/laser_amp_M,linestyle='--',c='g',label='filter field envelope',linewidth=1)
    ax2.plot(grid_x_axis/laser_lambda_M,-Electric_Field_filter_envelope/laser_amp_M,linestyle='--',c='g',linewidth=1)
    ax2.set_ylabel('E_filter(E0)', color='blue')
    ax2.set_ylim(-Electric_Field_filter_envelope_max/laser_amp_M,Electric_Field_filter_envelope_max/laser_amp_M)
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.legend(handles=[line1, line2,line3])
    #plt.xlim((grid_x_axis[Electric_Field_filter_envelope_max_id]-10*Electric_Field_filter_envelope_peak_width)/laser_lambda_M,(grid_x_axis[Electric_Field_filter_envelope_max_id]+10*Electric_Field_filter_envelope_peak_width)/laser_lambda_M)
    figure.tight_layout()
    figure.savefig(os.path.join(working_dir,'Electric_Field_filter_both_%s.png' %(name)))
    figure.clf()
    return {
        'Electric_Field_filter':Electric_Field_filter,
        'Electric_Field_filter_spectrum':Electric_Field_filter_spectrum,
        'Electric_Field_filter_envelope':Electric_Field_filter_envelope,
    }


def get_envelope(Electric_Field:np.ndarray,direction=1,name=''):
    """
        direction: direction>0 means the field travels in +x direction. direction<0 means the field travels in -x direction
    """
    assert direction!=0
    Electric_Field_continuation=continuation_field(Electric_Field,n_continuation_x=n_continuation_x)
    Electric_Field_analytic=hilbert(Electric_Field_continuation)[grid_center_mask]
    Electric_Field_envelope=np.abs(Electric_Field_analytic)
    Electric_Field_analytic_phase=np.sign(direction)*np.angle(Electric_Field_analytic)
    Electric_Field_envelope_square=np.square(Electric_Field_envelope)
    Electric_Field_envelope_x_1_moment=np.average(a=grid_x,weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_2_moment=np.average(a=np.square(grid_x),weights=Electric_Field_envelope_square)
    Electric_Field_envelope_x_std=np.sqrt(Electric_Field_envelope_x_2_moment-np.square(Electric_Field_envelope_x_1_moment))
    print('Lab frame average position of the envelope: %fλ0' %(Electric_Field_envelope_x_1_moment/laser_lambda_M))
    print('Lab frame FWHM obtained from envelope_x_std: %ffs'%(Electric_Field_envelope_x_std*2*math.sqrt(2*math.log(2))*math.cos(theta_rad)/C.speed_of_light/C.femto))
    Electric_Field_envelope_max=np.max(Electric_Field_envelope[1:-1])
    Electric_Field_envelope_max_id=np.where(Electric_Field_envelope==Electric_Field_envelope_max)[0].item()
    print(Electric_Field_envelope_max_id)
    print(Electric_Field_envelope_max/laser_amp_M)
    Electric_Field_envelope_max_phase=Electric_Field_analytic_phase[Electric_Field_envelope_max_id]
    print('Phase at the peak: %fπ' %(Electric_Field_envelope_max_phase/np.pi))
    Electric_Field_analytic_phase_unwrap=np.unwrap(Electric_Field_analytic_phase,period=2*np.pi)
    Electric_Field_analytic_phase_unwrap=Electric_Field_analytic_phase_unwrap-Electric_Field_analytic_phase_unwrap[Electric_Field_envelope_max_id]+Electric_Field_envelope_max_phase   #Phase relative to the peak. Keep the phase at the peak
    Electric_Field_envelope_prak=peak_widths(x=Electric_Field_envelope_square,peaks=[Electric_Field_envelope_max_id],rel_height=0.5)
    Electric_Field_envelope_peak_width=d_x*Electric_Field_envelope_prak[0].item()   #unit: m. In moving frame
    print('Lab frame FWHM obtained from envelope:%ffs' %(Electric_Field_envelope_peak_width*math.cos(theta_rad)/C.speed_of_light/C.femto))
    print('peak at x_L/λ0=%f' %(grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M))
    

    
    
    plt.plot(grid_x_axis/laser_lambda_M,continuation_field(Electric_Field,n_continuation_x=n_field_x)/laser_amp_M,c='b',label='field',linewidth=2)
    plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_envelope/laser_amp_M,linestyle='--',c='g',label='envelope',linewidth=1)
    plt.plot(grid_x_axis/laser_lambda_M,-Electric_Field_envelope/laser_amp_M,linestyle='--',c='g',linewidth=1)
    #plt.xlim((grid_x_axis[Electric_Field_envelope_max_id]-2*Electric_Field_envelope_peak_width)/laser_lambda_M,(grid_x_axis[Electric_Field_envelope_max_id]+2*Electric_Field_envelope_peak_width)/laser_lambda_M)
    #plt.xlim(grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M-4,grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M+4)
    #plt.ylim(-Electric_Field_envelope_max/laser_amp_M,Electric_Field_envelope_max/laser_amp_M)
    plt.ylim(-1,1)
    plt.legend()
    plt.xlabel(xlabel='x_L/λ0')
    plt.ylabel(ylabel='E(E0)')
    plt.title(label='Field_%s' %(name))
    plt.savefig(os.path.join(working_dir,'Electric_Field_envelope_%s.png' %(name)))
    plt.clf()
    return {
        'Electric_Field_envelope':Electric_Field_envelope,
        'Electric_Field_envelope_max':Electric_Field_envelope_max,
        'Electric_Field_envelope_max_id':Electric_Field_envelope_max_id,
        'Electric_Field_envelope_max_position':grid_x_axis[Electric_Field_envelope_max_id],
        'Electric_Field_envelope_max_phase':Electric_Field_envelope_max_phase,
        'Electric_Field_envelope_peak_width':Electric_Field_envelope_peak_width,
    }
    
    


def get_energy_flux(Electric_Field:np.ndarray,Magnetic_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,)
    assert Magnetic_Field.shape==(n_field_x,)
    S=Electric_Field*Magnetic_Field/C.mu_0
    S_continuation=continuation_field(S)
    S_spectrum=np.fft.fftshift(np.fft.fft(S_continuation))
    S_spectrum_square=np.square(np.abs(S_spectrum))
    S_spectrum_filter=S_spectrum*(freq_radius/laser_f0<1)
    S_average=np.real(np.fft.ifft(np.fft.ifftshift(S_spectrum_filter)))[0:n_field_x]
    S_average_max=np.max(np.abs(S_average))
    print(S_average_max/laser_S0_M)
    print(np.where(np.abs(S_average)==S_average_max))
    return S_average
    #plt.plot(grid_x_axis/laser_lambda_M,S/laser_S0_M,c='b',label='S',linewidth=1)
    plt.plot(grid_x_axis/laser_lambda_M,S_average/laser_S0_M,c='r',label='<S>',linewidth=2)
    #plt.xlim(6.6,7.4)
    plt.ylim(-0.002,0.002)
    plt.legend()
    plt.xlabel(xlabel='x_L/λ0')
    plt.ylabel(ylabel='S(S0)')
    plt.title(label='Energy_flux_Sx')
    plt.savefig(os.path.join(working_dir,'Energy_flux_Sx_%s.png' %(name)))
    plt.clf()
    
    return S_average

def get_power(Electric_Field_spectrum:np.ndarray,name=''):
    assert Electric_Field_spectrum.shape==(n_continuation_x,)
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    m, c = curve_fit(linear, np.log(freq_x[freq_mask]), np.log(Electric_Field_spectrum_square[freq_mask]))[0]
    print(m,c)
    Electric_Field_spectrum_square_fit=np.exp(linear(np.log(freq_x),m,c))
    plt.loglog(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='spectrum')
    plt.loglog(freq_x[n_continuation_x//2:]/laser_f0_M,Electric_Field_spectrum_square_fit[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='fit, -p=%0.2f'%(m),linestyle='--')
    plt.xlim(0.8,200)
    plt.ylim(1e-8,1)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.legend()
    plt.title(label='Spectrum')
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_square_fit_%s.png' %(name)))
    plt.clf()

def get_phase(Electric_Field:np.ndarray,harmonic_order=1,name=''):
    assert Electric_Field.shape==(n_field_x,)
    kn_M=harmonic_order*laser_k0_M
    Electric_Field_spectrum,Electric_Field_spectrum_square=get_spectrum(Electric_Field=Electric_Field,name=name)
    harmonic_order_th_mask=np.arange(n_continuation_x)[round(n_continuation_x/2+(harmonic_order-0.5)*space_length_lambda_M):round(n_continuation_x/2+(harmonic_order+0.5)*space_length_lambda_M)]
    freq_x_order_th=freq_x[harmonic_order_th_mask]
    k_x_order_th=k_x[harmonic_order_th_mask]
    Electric_Field_spectrum_order_th=Electric_Field_spectrum[harmonic_order_th_mask]
    Electric_Field_spectrum_order_th_square=np.square(np.abs(Electric_Field_spectrum_order_th))
    Electric_Field_spectrum_phase=np.unwrap(np.angle(Electric_Field_spectrum),period=2*np.pi)
    Electric_Field_spectrum_phase_order_th=Electric_Field_spectrum_phase[harmonic_order_th_mask]
    Electric_Field_spectrum_group_delay=-np.gradient(Electric_Field_spectrum_phase,k_x)   #dφ/dk,unit: m
    Electric_Field_spectrum_group_delay_order_th=Electric_Field_spectrum_group_delay[harmonic_order_th_mask]
    Electric_Field_spectrum_order_th_square_max=np.max(Electric_Field_spectrum_order_th_square)
    Electric_Field_spectrum_order_th_square_max_id=np.argmax(Electric_Field_spectrum_order_th_square)
    print(freq_x_order_th[Electric_Field_spectrum_order_th_square_max_id]/laser_f0_M)
    Electric_Field_spectrum_order_th_square_peak=peak_widths(x=Electric_Field_spectrum_order_th_square,peaks=[Electric_Field_spectrum_order_th_square_max_id],rel_height=0.1)
    Electric_Field_spectrum_group_delay_order_th_coefficients=np.polyfit(
        x=k_x_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        y=Electric_Field_spectrum_group_delay_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        deg=2
        )
    Electric_Field_spectrum_group_delay_order_th_fit=np.poly1d(c_or_r=Electric_Field_spectrum_group_delay_order_th_coefficients,r=False)(k_x_order_th)
    Electric_Field_spectrum_group_delay_order_th_average=np.average(
        a=Electric_Field_spectrum_group_delay_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())],
        #weights=Electric_Field_spectrum_square[round(Electric_Field_spectrum_order_th_square_peak[2].item()):round(Electric_Field_spectrum_order_th_square_peak[3].item())]
        )   #unit: m
    Electric_Field_spectrum_group_delay_order_th_0=Electric_Field_spectrum_group_delay_order_th[round(0.5*space_length_lambda_M)]
    print('Lab frame %d th harmonic at position %fλ0(average)' %(harmonic_order,vacuum_length_x_lambda_M/2+Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M))
    print('Lab frame %d th harmonic at position %fλ0(center)' %(harmonic_order,vacuum_length_x_lambda_M/2+Electric_Field_spectrum_group_delay_order_th_0/laser_lambda_M))
    print('Lab frame %d th harmonic at position %fλ0(peak)' %(harmonic_order,vacuum_length_x_lambda_M/2+Electric_Field_spectrum_group_delay_order_th[Electric_Field_spectrum_order_th_square_max_id]/laser_lambda_M))
    Electric_Field_spectrum_intrinsic_phase=Electric_Field_spectrum_phase+k_x*Electric_Field_spectrum_group_delay_order_th_0
    Electric_Field_spectrum_intrinsic_phase_order_th=Electric_Field_spectrum_intrinsic_phase[harmonic_order_th_mask]
    Electric_Field_spectrum_intrinsic_phase_order_th_0=Electric_Field_spectrum_intrinsic_phase_order_th[round(0.5*space_length_lambda_M)]
    print('group_delay dφ(k)/dk=α*(k/kn)^2+β*(k/kn)+x0, kn=%d*k0' %(harmonic_order))
    print('Lab frame x0=%fλ0' %(Electric_Field_spectrum_group_delay_order_th_coefficients[2]/laser_lambda_M))
    print('β=%f' %(Electric_Field_spectrum_group_delay_order_th_coefficients[1]*kn_M))
    print('α=%f' %(Electric_Field_spectrum_group_delay_order_th_coefficients[0]*kn_M**2))
    print('φ_intrinsic_kn+-φ_intrinsic_kn-=%fπ' %((Electric_Field_spectrum_intrinsic_phase_order_th[round(Electric_Field_spectrum_order_th_square_peak[3].item())]-Electric_Field_spectrum_intrinsic_phase_order_th[round(Electric_Field_spectrum_order_th_square_peak[2].item())])/np.pi))
    
    figure, ax1 = plt.subplots()
    plt.xlabel(xlabel='k/k0')
    ax2 = ax1.twinx() 
    line1,=ax1.semilogy(freq_x_order_th/laser_f0_M,Electric_Field_spectrum_order_th_square/laser_spectrum_peak_M**2,label='Spectrum', color='red')
    ax1.set_ylim(1e-6,1.1*Electric_Field_spectrum_order_th_square_max/laser_spectrum_peak_M**2)
    ax1.set_ylabel('I(k)/I0',color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    line2,=ax2.plot(freq_x_order_th/laser_f0_M,Electric_Field_spectrum_intrinsic_phase_order_th/np.pi,label='Phase', color='blue')
    ax2.set_ylim(Electric_Field_spectrum_intrinsic_phase_order_th_0/np.pi-1,Electric_Field_spectrum_intrinsic_phase_order_th_0/np.pi+1)
    ax2.set_ylabel('φ/π', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.tight_layout()
    plt.legend(handles=[line1,line2])
    plt.title('Phase of the %d harmonic' %(harmonic_order))
    plt.xlim(harmonic_order-0.1,harmonic_order+0.1)
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_phase_%s.png' %(name)))
    plt.clf()
    figure, ax1 = plt.subplots()
    plt.xlabel(xlabel='k/k0')
    ax2 = ax1.twinx() 
    line1,=ax1.semilogy(freq_x_order_th/laser_f0_M,Electric_Field_spectrum_order_th_square/laser_spectrum_peak_M**2,label='Spectrum', color='red')
    ax1.set_ylim(1e-6,1.1*Electric_Field_spectrum_order_th_square_max/laser_spectrum_peak_M**2)
    ax1.set_ylabel('I(k)/I0',color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    line2,=ax2.plot(freq_x_order_th/laser_f0_M,Electric_Field_spectrum_group_delay_order_th/laser_lambda_M,label='group delay', color='blue')
    line3,=ax2.plot(freq_x_order_th/laser_f0_M,Electric_Field_spectrum_group_delay_order_th_fit/laser_lambda_M,linestyle='--',label='group delay fit')
    ax2.set_ylim((Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M-10,Electric_Field_spectrum_group_delay_order_th_average/laser_lambda_M+10))
    ax2.set_ylabel('(dφ/dk)/λ0', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.tight_layout()
    plt.legend(handles=[line1,line2,line3])
    plt.title('Group delay of the %d harmonic' %(harmonic_order))
    plt.xlim(harmonic_order-0.1,harmonic_order+0.1)
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_group_delay_%s.png' %(name)))
    plt.clf()
    return (Electric_Field_spectrum_group_delay_order_th_average/laser_lambda-continuation_length_lambda)*math.cos(theta_rad)
    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    ax1.semilogy(freq_x/laser_f0_M,Electric_Field_spectrum_square/laser_spectrum_peak_M**2,label='Spectrum', color='red')
    ax1.set_ylim(1e-6,1.1)
    ax1.set_ylabel('I(k)/I0',color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.plot(freq_x/laser_f0_M,Electric_Field_spectrum_intrinsic_phase/np.pi,label='Phase', color='blue')
    ax2.set_ylim(Electric_Field_spectrum_intrinsic_phase_order_th_0/np.pi-10,Electric_Field_spectrum_intrinsic_phase_order_th_0/np.pi+10000)
    ax2.set_ylabel('φ/π', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.tight_layout()
    plt.legend()
    plt.title('Phase')
    plt.xlim(0,5)
    plt.xlabel(xlabel='k/k0')
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_phase_all_%s.png' %(name)))
    plt.clf()
    figure, ax1 = plt.subplots()
    ax2 = ax1.twinx() 
    ax1.semilogy(freq_x/laser_f0_M,Electric_Field_spectrum_square/laser_spectrum_peak_M**2,label='Spectrum', color='red')
    ax1.set_ylim(1e-6,1.1)
    ax1.set_ylabel('I(k)/I0',color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2.plot(freq_x[Electric_Field_spectrum_square/laser_spectrum_peak_M**2>1e-3]/laser_f0_M,Electric_Field_spectrum_group_delay[Electric_Field_spectrum_square/laser_spectrum_peak_M**2>1e-3]/laser_lambda,label='group delay', color='blue')
    ax2.set_ylim((Electric_Field_spectrum_group_delay_order_th_average/laser_lambda-5,Electric_Field_spectrum_group_delay_order_th_average/laser_lambda+5))
    ax2.set_ylabel('(dφ/dk)/λ0', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.tight_layout()
    plt.legend()
    plt.title('Group delay')
    plt.xlim(0,5)
    plt.xlabel(xlabel='k/k0')
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_group_delay_all_%s.png' %(name)))
    plt.clf()



    
    
def get_spectrum(Electric_Field:np.ndarray,name=''):
    assert Electric_Field.shape==(n_field_x,) or Electric_Field.shape==(n_continuation_x,)
    if Electric_Field.shape==(n_field_x,):
        Electric_Field_continuation=continuation_field(Electric_Field)
    else:
        Electric_Field_continuation=Electric_Field
    Electric_Field_spectrum=np.fft.fftshift(np.fft.fft(a=fftshift(Electric_Field_continuation),n=n_continuation_x,axis=0))*d_x
    Electric_Field_spectrum_square=np.square(np.abs(Electric_Field_spectrum))
    
    print(freq_x_axis[np.where(Electric_Field_spectrum_square==np.max(Electric_Field_spectrum_square))[0]]/laser_f0_M)

    print('Total energy %e' %(np.sum(Electric_Field_spectrum_square)*d_f/laser_energy_M))
    #pd.DataFrame(data=Electric_Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,index=freq_x_axis[n_continuation_x//2:]/laser_f0_M).to_hdf(path_or_buf=os.path.join(working_dir,'spectrum.hdf5'),mode='a',key='Electric_Field_spectrum_%s' %(name))
    #return Electric_Field_spectrum,Electric_Field_spectrum_square
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2)
    plt.xlim(0,10)
    plt.ylim(1e-6,1.1)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.title(label='Spectrum')
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_%s.png' %(name)))
    plt.clf()
    return Electric_Field_spectrum,Electric_Field_spectrum_square

    

    

def output_field(Electric_Field:np.ndarray,grid_axis:np.ndarray,name=''):
    """
        Electric_Field (np.ndarray): Fields on the grid in the moving frame with coordinate grid_x_L_axis=grid_x_axis*cos(θ)
        grid_axis (np.ndarray): Grid in the lab frame.
    """
    assert Electric_Field.shape==(n_field_x,)
    Electric_Field_L=Electric_Field/math.cos(theta_rad)
    Electric_Field_L_interp_func = interp1d(x=grid_x_L_axis, y=Electric_Field_L, kind='cubic', fill_value=0)
    Electric_Field_L_output=Electric_Field_L_interp_func(grid_axis)
    fields_ds=xr.Dataset(
        data_vars={
            name:(["x"], Electric_Field_L_output),
            },
        coords={'x':(["x"], grid_axis)}
        )
    fields_ds.to_netcdf(path=os.path.join(working_dir,'%s.nc' %(name)))
    return Electric_Field_L_output
    


def two_pulse_spectrum(Electric_Field_1:np.ndarray,Electric_Field_2:np.ndarray):
    Electric_Field_1_continuation=continuation_field(Electric_Field_1)
    Electric_Field_2_continuation=continuation_field(Electric_Field_2)
    Electric_Field_concatenate=np.concatenate((Electric_Field_1,Electric_Field_2),axis=0)
    Electric_Field_concatenate_continuation=continuation_field(Electric_Field_concatenate)
    _,Electric_Field_1_spectrum_square=get_spectrum(Electric_Field_1_continuation,name='Electric_Field_1')
    _,Electric_Field_2_spectrum_square=get_spectrum(Electric_Field_2_continuation,name='Electric_Field_2')
    _,Electric_Field_concatenate_spectrum_square=get_spectrum(Electric_Field_concatenate_continuation,name='Electric_Field_concatenate')
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_1_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='1',linestyle='--',c='b',linewidth=1)
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_2_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='2',linestyle='--',c='g',linewidth=1)
    plt.semilogy(freq_x_axis[n_continuation_x//2:]/laser_f0_M,Electric_Field_concatenate_spectrum_square[n_continuation_x//2:]/laser_spectrum_peak_M**2,label='concatenate',c='r')
    plt.xlim(5,10)
    plt.ylim(1e-6,1e-2)
    plt.xlabel(xlabel='k/k0')
    plt.ylabel(ylabel='I(k)/I0')
    plt.title(label='Spectrum')
    plt.legend()
    plt.savefig(os.path.join(working_dir,'Electric_Field_spectrum_continuation.png'))
    plt.clf()

i=110
sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
Electric_Field_Ey_left=sdf.__dict__['Electric_Field_Ey'].data[:n_field_x]
Electric_Field_Ey_left=reverse_field(Electric_Field_Ey_left)
get_spectrum(Electric_Field_Ey_left,name='%0.4d_left' %i)
Electric_Field_Ey_left_2_plus_envelope=filter_field(Electric_Field_Ey_left,(1.5,500),name='%0.4d_left_2+'%i)['Electric_Field_filter_envelope']
Electric_Field_Ey_left_1_envelope=filter_field(Electric_Field_Ey_left,(0.5,1.5),name='%0.4d_1'%i)['Electric_Field_filter_envelope']
Electric_Field_Ey_left_2_envelope=filter_field(Electric_Field_Ey_left,(1.5,2.5),name='%0.4d_2'%i)['Electric_Field_filter_envelope']
Electric_Field_Ey_left_3_envelope=filter_field(Electric_Field_Ey_left,(2.5,3.5),name='%0.4d_3'%i)['Electric_Field_filter_envelope']
Electric_Field_Ey_left_10_envelope=filter_field(Electric_Field_Ey_left,(9.5,10.5),name='%0.4d_10'%i)['Electric_Field_filter_envelope']
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_Ey_left_1_envelope/np.max(Electric_Field_Ey_left_1_envelope),label='1')
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_Ey_left_2_envelope/np.max(Electric_Field_Ey_left_2_envelope),label='2')
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_Ey_left_10_envelope/np.max(Electric_Field_Ey_left_10_envelope),label='10')
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_Ey_left_3_envelope/np.max(Electric_Field_Ey_left_3_envelope),label='3')
plt.plot(grid_x_axis/laser_lambda_M,Electric_Field_Ey_left_2_plus_envelope/np.max(Electric_Field_Ey_left_2_plus_envelope),label='[2,1000]')
plt.xlim(20,50)
plt.ylim(0,1.1)
plt.legend()
plt.xlabel(xlabel='x_L/λ0')
plt.ylabel(ylabel='E(norm)')
plt.title(label='Envelope')
plt.savefig(os.path.join(working_dir,'Electric_Field_envelope.png'))
plt.clf()
exit(0)




reflection_rotate_field=-read_nc(nc_name='/scratch/gpfs/MIKHAILOVA/zl8336/45thin/2D/rotate_field/Electric_Field_rotate_0023_reflection_250cpl.nc',key_name_list=['Electric_Field_transversal'])['Electric_Field_transversal'][:,2000]
reflection_rotate_target=read_nc(nc_name='/scratch/gpfs/MIKHAILOVA/zl8336/45thin/2D/rotate_target/Electric_Field_rotate_0031_reflection_250cpl.nc',key_name_list=['Electric_Field_transversal'])['Electric_Field_transversal'][:,2000]
reflection_rotate_field_2=filter_field(reflection_rotate_field,(2.5,100))[0]
reflection_rotate_target_2=filter_field(reflection_rotate_target,(2.5,100))[0]
reflection_rotate_field_2_envelope=get_envelope(reflection_rotate_field_2,name='reflection_rotate_field')['Electric_Field_envelope']
reflection_rotate_target_2_envelope=get_envelope(reflection_rotate_target_2,name='reflection_rotate_target')['Electric_Field_envelope']
plt.plot(grid_x_axis/laser_lambda_M,reflection_rotate_field_2/laser_amp_M,c='b',label='rotate_field',linewidth=1)
plt.plot(grid_x_axis/laser_lambda_M,reflection_rotate_target_2/laser_amp_M,c='r',label='rotate_target',linewidth=1)
plt.plot(grid_x_axis/laser_lambda_M,reflection_rotate_field_2_envelope/laser_amp_M,linestyle='--',c='b',label='rotate_field_envelope',linewidth=1)
plt.plot(grid_x_axis/laser_lambda_M,reflection_rotate_target_2_envelope/laser_amp_M,linestyle='--',c='r',label='rotate_target_envelope',linewidth=1)
#plt.xlim((grid_x_axis[Electric_Field_envelope_max_id]-2*Electric_Field_envelope_peak_width)/laser_lambda_M,(grid_x_axis[Electric_Field_envelope_max_id]+2*Electric_Field_envelope_peak_width)/laser_lambda_M)
#plt.xlim(grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M-4,grid_x_axis[Electric_Field_envelope_max_id]/laser_lambda_M+4)
#plt.ylim(-Electric_Field_envelope_max/laser_amp_M,Electric_Field_envelope_max/laser_amp_M)
plt.ylim(-0.5,0.5)
plt.legend()
plt.xlabel(xlabel='x_L/λ0')
plt.ylabel(ylabel='E(E0)')
plt.title(label='Field centerline')
plt.savefig(os.path.join(working_dir,'reflection_envelop_3.png'))
plt.clf()
exit(0)

i=23
sdf=sdf_helper.getdata(fname=os.path.join(working_dir,'%0.4d.sdf' %(i)))
Electric_Field_Ey=sdf.__dict__['Electric_Field_Ey'].data[0:n_field_x]
get_spectrum(Electric_Field_Ey,name='%0.4d_reflection' %(i))
filter_field(Electric_Field_Ey,(0.5,1.5))
filter_field(Electric_Field_Ey,(1.5,2.5))
exit(0)
Electric_Field_Ey_reflection_filter,_,_=filter_field(Electric_Field_Ey[:n_field_x],(5,200),name='reflection_5+_%0.4d' %(i))
Electric_Field_Ey_reflection_filter=reverse_field(Electric_Field_Ey_reflection_filter)
Electric_Field_Ey_transmition_filter,_,_=filter_field(Electric_Field_Ey[n_field_x:],(5,200),name='transmition_5+_%0.4d' %(i))
x_r=np.linspace(7*laser_lambda,13*laser_lambda,num=6*cells_per_lambda,endpoint=False)
output_field(Electric_Field_Ey_reflection_filter,x_r,name='Electric_Field_Ey_reflection_filter')
x_t=np.linspace(6*laser_lambda,12*laser_lambda,num=6*cells_per_lambda,endpoint=False)
output_field(Electric_Field_Ey_transmition_filter,x_t,name='Electric_Field_Ey_transmition_filter')


exit(0)

i=23
fields=xr.open_dataset('/scratch/gpfs/MIKHAILOVA/zl8336/45thin/1D/Electric_Field_Ey_reflection_filter.nc')
Electric_Field_Ey_reflection_filter=fields['Electric_Field_Ey_reflection_filter'].to_numpy()
get_envelope(Electric_Field_Ey_reflection_filter,'Electric_Field_Ey_reflection_filter_nc_%0.4d' %(i))
fields=xr.open_dataset('/scratch/gpfs/MIKHAILOVA/zl8336/45thin/1D/Electric_Field_Ey_transmition_filter.nc')
Electric_Field_Ey_reflection_filter=fields['Electric_Field_Ey_transmition_filter'].to_numpy()
get_envelope(Electric_Field_Ey_reflection_filter,'Electric_Field_Ey_transmition_filter_nc_%0.4d' %(i))
exit(0)










'incident'
'reflection,reflected'
'transmission,transmitted'