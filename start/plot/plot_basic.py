import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def savefig(fig=None,fig_path=''):
    if fig is None:
        fig=plt.gcf()
    plt.savefig(fig_path)
    plt.close(fig)
    plt.clf()
    print(fig_path)
    return fig_path



def phase_amp_to_rgb(
    phase:np.ndarray,amplitude:np.ndarray,amplitude_max=None
    ):
    """
    Convert phase and amplitude to rgb image  
    parameters:
    phase: ND array, representing phase data (range should be [-π, π])
    amplitude: ND array, representing amplitude data
    returns:
    rgb_image: (N+1)D array, shape (*shape, 3), representing RGB image
    """
    phase=np.asarray(phase)
    amplitude=np.asarray(amplitude) 
    shape=phase.shape
    assert amplitude.shape==shape, "Phase and amplitude must have the same shape."
    amplitude=np.abs(amplitude)
    phase_normalized=np.mod((phase+np.pi)/(2*np.pi), 1)   #normalize to [0, 1], ['-π', '-π/2', '0', 'π/2', 'π']->[0, 0.25, 0.5, 0.75, 1]
    if amplitude_max is None:
        amplitude_max=np.max(amplitude)
    amplitude_normalized=np.where(amplitude>amplitude_max, 1.0, amplitude/amplitude_max)   #normalize to [0, 1]
    h=phase_normalized  #phase as hue, shape (*shape)
    s=np.ones_like(h)  #saturation fixed to 1, shape (*shape)
    v=amplitude_normalized  #amplitude as value, shape (*shape)
    hsv_image=np.stack((h,s,v),axis=-1)   #shape (*shape, 3)
    rgb_image=hsv_to_rgb(hsv_image)  #shape (*shape, 3)
    return rgb_image 
    

def Plot_complex_field_2D(
    A:np.ndarray=None, 
    phase:np.ndarray=None,amplitude:np.ndarray=None,
    A_max=None,
    x_axis=None,y_axis=None,
    xlabel='x',ylabel='y',label='',
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    plot_polar_colorbar=False,working_dir='.'
    ):
    """
    绘制二维复数场，并使用极坐标colorbar同时展示相位和振幅
    All variables should be normalized in advance.
    参数:
    A: 二维复数数组
    """
    assert (A is not None) or (phase is not None and amplitude is not None), "Either A or both phase and amplitude must be provided."
    if A is not None:
        A=np.asarray(A)
        phase = np.angle(A)  # 范围 [-π, π]
        amplitude = np.abs(A)
        Nx,Ny=phase.shape
    if x_axis is None:
        x_axis=np.arange(Nx)
    if y_axis is None:
        y_axis=np.arange(Ny)
    assert phase.shape==(Nx,Ny)
    assert amplitude.shape==(Nx,Ny)
    # 计算相位和振幅
    norm=Normalize(vmin=0, vmax=1)
    rgb_image = phase_amp_to_rgb(phase, amplitude, amplitude_max=A_max)  #shape (Nx, Ny, 3)
    sm = ScalarMappable(norm=norm, cmap='hsv')
    sm.set_array([])
    if xmin is None:
        xmin=x_axis[0]
    if xmax is None:
        xmax=x_axis[-1]
    if ymin is None:
        ymin=y_axis[0]
    if ymax is None:
        ymax=y_axis[-1]
    # 创建图形
    fig,ax_main = plt.subplots()
    pcm = ax_main.pcolormesh(x_axis,y_axis,np.zeros((Ny,Nx)),color=rgb_image.transpose(1,0,2).reshape(-1, 3),shading='auto')
    ax_main.set_aspect('equal')
    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_main.set_xlim(xmin,xmax)
    ax_main.set_ylim(ymin,ymax)
    ax_main.set_title(label, fontsize=12)
    ax_main.grid(True, alpha=0.2, linestyle='--')
    ax_cbar=plt.colorbar(mappable=sm,ax=ax_main).ax
    phase_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_tick_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax_cbar.set_yticks(phase_ticks)
    ax_cbar.set_yticklabels(phase_tick_labels, fontsize=9)
    ax_cbar.set_ylabel('phase (rad)', fontsize=10)
    savefig(fig=fig,fig_path=os.path.join(working_dir,f'Complex_field_{label}.png'))
    if plot_polar_colorbar:
        plot_polar_hsv_colorbar(label=label)
    

def plot_polar_hsv_colorbar(label=''):
    Nr=100   #nomber of radial divisions
    Nt=200   #number of angular divisions

    r_axis = np.linspace(0, 1, Nr)  # 半径表示振幅 [0, 1]
    theta_axis = np.linspace(0, 2*np.pi, Nt,endpoint=False)  # 角度表示相位 [0, 2π]
    
    r, theta = np.meshgrid(r_axis, theta_axis,indexing='ij')
    rgb_image = phase_amp_to_rgb(theta, r)  #shape (Nr, Nt, 3)
    fig = plt.figure()
    ax_polar = fig.add_subplot(111, projection='polar')
    
    pcm = ax_polar.pcolormesh(theta_axis, r_axis,np.zeros((Nr,Nt)),shading='auto', color=rgb_image.reshape(-1, 3))
    
    # 设置极坐标图属性
    ax_polar.set_theta_zero_location('E')  # 0度在右侧
    ax_polar.set_theta_direction(1)  # 角度逆时针增加
    ax_polar.set_rlim(0, 1)
    
    # 设置相位刻度 (角度)
    phase_ticks = np.linspace(0, 2*np.pi, 8, endpoint=False)
    phase_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
    ax_polar.set_xticks(phase_ticks)
    ax_polar.set_xticklabels(phase_labels, fontsize=9)
    
    # 设置振幅刻度 (半径)
    amp_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    amp_labels = ['0', '0.25', '0.5', '0.75', f'{label}=1.0']
    ax_polar.set_yticks(amp_ticks)
    ax_polar.set_yticklabels(amp_labels, fontsize=9)
    
    # 添加极坐标colorbar标题和标签
    ax_polar.set_title('phase-amplitude', fontsize=12, pad=20)
    
    # 添加参考线
    ax_polar.plot([0, 0], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 0°参考线
    ax_polar.plot([np.pi/2, np.pi/2], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 90°参考线
    ax_polar.plot([np.pi, np.pi], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 180°参考线
    ax_polar.plot([3*np.pi/2, 3*np.pi/2], [0, 1], 'w--', alpha=0.5, linewidth=0.8)  # 270°参考线
    
    # 添加圆形网格
    for r_val in amp_ticks:
        circle = Circle((0, 0), r_val, transform=ax_polar.transData._b, 
                       fill=False, edgecolor='white', alpha=0.3, linewidth=0.5)
        ax_polar.add_artist(circle)
    norm=Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='hsv')
    sm.set_array([])
    ax_cbar=plt.colorbar(mappable=sm,ax=ax_polar).ax
    phase_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_tick_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax_cbar.set_yticks(phase_ticks)
    ax_cbar.set_yticklabels(phase_tick_labels, fontsize=9)
    ax_cbar.set_ylabel('phase (rad)', fontsize=10)
    savefig(fig=fig,fig_path=os.path.join(working_dir,f'Polar_colorbar_{label}.png'))

def Plot_complex_field_3D(
    A:np.ndarray=None, 
    phase:np.ndarray=None,amplitude:np.ndarray=None,
    A_max=None,
    x_axis=None,y_axis=None,z_axis=None,
    xlabel='x',ylabel='y',zlabel='z',label='',
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,zmin:Optional[float]=None,zmax:Optional[float]=None,
    ):
    pass  # To be implemented in the future


working_dir="/scratch/gpfs/MIKHAILOVA/zl8336/start/plot"

