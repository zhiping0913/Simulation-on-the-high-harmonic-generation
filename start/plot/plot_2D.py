import sys
sys.path.append('/scratch/gpfs/MIKHAILOVA/zl8336/start')
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize,XKCD_COLORS
from plot.plot_basic import savefig
from plot.plot_1D import plot_multiple_1D_fields
from Spectral_Maxwell.pretreat_fields import get_closest_coordinate_id

working_dir=''

def generate_side_panel_axes(
    ax_main_height=8.0,ax_main_width=8.0,side_panel_size=3.0,ax_cbar_width = 0.3,
    generate_ax_panel_top=False,generate_ax_panel_right=False,generate_ax_panel_bottom=False,generate_ax_panel_left=False,
    generate_ax_legend=False,ax_cbar_num=1,
    ):
    """_summary_
    Args:
        ax_main_height (float, optional): _description_. Defaults to 8. unit: inch.
        ax_main_width (float, optional): _description_. Defaults to 8. unit: inch.
        side_panel_size (float, optional): _description_. Defaults to 3. unit: inch.
        ax_cbar_width (float, optional): _description_. Defaults to 0.3. unit: inch.
        generate_ax_panel_top (bool, optional): _description_. Defaults to False.
        generate_ax_panel_right (bool, optional): _description_. Defaults to False.
        generate_ax_panel_bottom (bool, optional): _description_. Defaults to False.
        generate_ax_panel_left (bool, optional): _description_. Defaults to False.
        ax_cbar_num (int, optional): _description_. Defaults to 1.
    """
    figure_base_left = 1.0   #unit: inch
    figure_base_right = 1.0   #unit: inch
    figure_base_bottom = 1.0   #unit: inch
    figure_base_top = 1.5   #unit: inch
    ax_cbar_gap = 0.5 #unit: inch
    ax_cbar_height = ax_main_height #unit: inch
    figure_width = figure_base_left + ax_main_width + generate_ax_panel_left*side_panel_size+generate_ax_panel_right*side_panel_size + ax_cbar_gap +ax_cbar_num * (ax_cbar_width+ax_cbar_gap) +figure_base_right #unit: inch
    figure_height = figure_base_bottom + ax_main_height + generate_ax_panel_top*side_panel_size+generate_ax_panel_bottom*side_panel_size +figure_base_top #unit: inch
    fig = plt.figure(figsize=(figure_width,figure_height),dpi=100)
    print('figure size=',fig.get_size_inches(),'inch')
    ax_main_left=figure_base_left+generate_ax_panel_left*side_panel_size   #unit: inch
    ax_main_bottom=figure_base_bottom+generate_ax_panel_bottom*side_panel_size   #unit: inch
    ax_main: plt.Axes = fig.add_axes([ax_main_left/figure_width, ax_main_bottom/figure_height, ax_main_width/figure_width, ax_main_height/figure_height])
    print('ax_main position=',ax_main.get_position())
    ax_dict={'fig':fig,'ax_main':ax_main}
    if generate_ax_panel_top:
        ax_panel_top_left = ax_main_left
        ax_panel_top_bottom = ax_main_bottom + ax_main_height
        ax_panel_top = fig.add_axes([ax_panel_top_left/figure_width, ax_panel_top_bottom/figure_height, ax_main_width/figure_width, side_panel_size/figure_height],sharex=ax_main)
        ax_dict['ax_panel_top']=ax_panel_top
    if generate_ax_panel_right:
        ax_panel_right_left = ax_main_left + ax_main_width
        ax_panel_right_bottom = ax_main_bottom
        ax_panel_right = fig.add_axes([ax_panel_right_left/figure_width, ax_panel_right_bottom/figure_height, side_panel_size/figure_width, ax_main_height/figure_height],sharey=ax_main)
        ax_dict['ax_panel_right']=ax_panel_right
    if generate_ax_panel_bottom:
        ax_panel_bottom_left = ax_main_left
        ax_panel_bottom_bottom = figure_base_bottom
        ax_panel_bottom = fig.add_axes([ax_panel_bottom_left/figure_width, ax_panel_bottom_bottom/figure_height, ax_main_width/figure_width, side_panel_size/figure_height],sharex=ax_main)
        ax_dict['ax_panel_bottom']=ax_panel_bottom
    if generate_ax_panel_left:
        ax_panel_left_left = figure_base_left
        ax_panel_left_bottom = ax_main_bottom
        ax_panel_left = fig.add_axes([ax_panel_left_left/figure_width, ax_panel_left_bottom/figure_height, side_panel_size/figure_width, ax_main_height/figure_height],sharey=ax_main)
        ax_dict['ax_panel_left']=ax_panel_left
    for i in range(ax_cbar_num):
        ax_cbar_i_left = ax_main_left + ax_main_width + generate_ax_panel_right*side_panel_size + ax_cbar_gap + i*(ax_cbar_width+ax_cbar_gap)
        ax_cbar_i_bottom = ax_main_bottom
        ax_cbar_i = fig.add_axes([ax_cbar_i_left/figure_width, ax_cbar_i_bottom/figure_height, ax_cbar_width/figure_width, ax_cbar_height/figure_height])
        ax_dict[f'ax_cbar_{i}']=ax_cbar_i
    if generate_ax_legend:
        ax_legend_left=ax_main_left + ax_main_width
        ax_legend_bottom=ax_main_bottom + ax_main_height
        ax_legend=fig.add_axes([ax_legend_left/figure_width,ax_legend_bottom/figure_height,side_panel_size/figure_width,side_panel_size/figure_height])
        ax_legend.set_facecolor((1, 1, 1, 0))
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        ax_dict['ax_legend']=ax_legend
    return ax_dict
    
def plot_2D_field_with_side_panel(
    field:np.ndarray,
    field_top_panel_dict:Optional[dict]=None,
    field_bottom_panel_dict:Optional[dict]=None,
    field_right_panel_dict:Optional[dict]=None,
    field_left_panel_dict:Optional[dict]=None,
    ax_dict:Optional[dict]=None,ax_cbar:Optional[plt.Axes]=None,
    x_coordinate=[0],y_coordinate=[0],
    vmin=-1,vmax=1,norm:Optional[Normalize]=None,cmap='seismic',alpha=1.0,aspect=1.0,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    zoom=1.0,plot_colorbar=True,
    return_fig=True,name='',working_dir='.',
    ):
    r"""_summary_
    All variables should be normalized in advance
    Args:
        field (np.ndarray): _description_
        field_top_panel_dict,field_bottom_panel_dict,field_right_panel_dict,field_left_panel_dict: 
        {
            'field_dict_list':[{'field':field,'linestyle':'-','label':None,'color':None},...] ,
            'axhline_dict_list':[{'y':y,'linestyle':'--','label':None,'color':None},...],
            'axvline_dict_list':[{'x':x,'linestyle':'--','label':None,'color':None},...],
            'vmin':float=vmin,
            'vmax':float=vmax,
            'scale':str | ScaleBase='linear',
            'label':str=label,
            'title':str=None,
            'plot_legend':bool=False,
        }
        ax_dict:
        {
            'fig':fig,
            'ax_main':ax_main,   #must be provided
            'ax_panel_top':ax_panel_top,   #can be None
            'ax_panel_right':ax_panel_right,   #can be None
            'ax_panel_bottom':ax_panel_bottom,   #can be None
            'ax_panel_left':ax_panel_left,   #can be None
            'ax_cbar_0':ax_cbar_0,   #can be None
        }
        x_coordinate (_type_, optional): _description_. Defaults to x_coordinate.
        y_coordinate (_type_, optional): _description_. Defaults to y_coordinate.
        vmin (_type_, optional): _description_. Defaults to -1.
        vmax (_type_, optional): _description_. Defaults to 1.
        cmap (str, optional): _description_. Defaults to 'seismic'.
        aspect: (str or float, optional): _description_. Defaults to 1 (equal aspect ratio).
        label (regexp, optional): _description_. Defaults to ''.
        return_fig (bool, optional): _description_. Defaults to True.
        name (str, optional): _description_. Defaults to ''.
        working_dir (str, optional): _description_. Defaults to '.'.
    Returns:
        _type_: _description_
    """
    
    step=round(1.0/zoom)
    field=np.asarray(field)
    x_coordinate=np.asarray(x_coordinate).flatten()
    y_coordinate=np.asarray(y_coordinate).flatten()
    n_x=x_coordinate.size
    n_y=y_coordinate.size
    assert field.shape==(n_x,n_y)
    if vmin is None:
        vmin=np.nanmin(field)
    if vmax is None:
        vmax=np.nanmax(field)
    if norm is None:
        norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
    if xmin is None:
        xmin=np.nanmin(x_coordinate)
    if xmax is None:
        xmax=np.nanmax(x_coordinate)
    if ymin is None:
        ymin=np.nanmin(y_coordinate)
    if ymax is None:
        ymax=np.nanmax(y_coordinate)
    l_x=xmax-xmin
    l_y=ymax-ymin
    if ax_dict is None:
        ax_main_height=6   #unit: inch
        ax_main_width=ax_main_height*l_x/(l_y*aspect)   #unit: inch
        ax_dict=generate_side_panel_axes(
            ax_main_height=ax_main_height,ax_main_width=ax_main_width,
            generate_ax_panel_top=field_top_panel_dict is not None,
            generate_ax_panel_right=field_right_panel_dict is not None,
            generate_ax_panel_bottom=field_bottom_panel_dict is not None,
            generate_ax_panel_left=field_left_panel_dict is not None,
            generate_ax_legend=True,
        )
    fig:plt.Figure=ax_dict['fig']
    ax_main:plt.Axes=ax_dict['ax_main']
    pcm = ax_main.pcolormesh(
        x_coordinate[::step], 
        y_coordinate[::step], 
        field[::step, ::step].T, 
        cmap=cmap, shading='auto',norm=norm, alpha=alpha,
        )
    ax_main.set_xlabel(xlabel,fontsize=15)
    ax_main.set_ylabel(ylabel,fontsize=15, rotation=0)
    ax_main.set_xlim(xmin,xmax)
    ax_main.set_ylim(ymin,ymax)
    ax_main.set_aspect(aspect)
    ax_main.grid(True, alpha=0.4)
    ax_dict['ax_main']=ax_main
    if field_top_panel_dict is not None:
        ax_panel_top:plt.Axes= ax_dict['ax_panel_top']
        ax_panel_top= plot_multiple_1D_fields(
            coordinate_direction='vertical',
            coordinate=x_coordinate,
            ax=ax_panel_top,
            field_dict_list=field_top_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_top_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_top_panel_dict.get('axvline_dict_list',None),
            xmin=xmin,xmax=xmax,
            ymin=field_top_panel_dict.get('vmin',vmin),ymax=field_top_panel_dict.get('vmax',vmax),
            xlabel=xlabel,ylabel=field_top_panel_dict.get('label',label),
            xscale='linear',yscale=field_top_panel_dict.get('scale','linear'),
            plot_legend=field_top_panel_dict.get('plot_legend',False),
            name=field_top_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        ax_panel_top.tick_params(axis='y', labelrotation=0,left=True,right=True,labelleft=True,labelright=True)
        ax_panel_top.tick_params(axis='x', labelrotation=0,top=True,bottom=False,labeltop=True,labelbottom=False)
        ax_dict['ax_panel_top']=ax_panel_top
    if field_bottom_panel_dict is not None:
        ax_panel_bottom:plt.Axes=ax_dict['ax_panel_bottom']
        ax_panel_bottom= plot_multiple_1D_fields(
            coordinate_direction='vertical',
            coordinate=x_coordinate,
            ax=ax_panel_bottom,
            field_dict_list=field_bottom_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_bottom_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_bottom_panel_dict.get('axvline_dict_list',None),
            xmin=xmin,xmax=xmax,
            ymin=field_bottom_panel_dict.get('vmin',vmin),ymax=field_bottom_panel_dict.get('vmax',vmax),
            xlabel=xlabel,ylabel=field_bottom_panel_dict.get('label',label),
            xscale='linear',yscale=field_bottom_panel_dict.get('scale','linear'),
            plot_legend=field_bottom_panel_dict.get('plot_legend',False),
        )['ax_main']
        ax_panel_bottom.tick_params(axis='y', labelrotation=0,left=True,right=True,labelleft=True,labelright=True)
        ax_panel_bottom.tick_params(axis='x', labelrotation=0,top=False,bottom=True,labeltop=False,labelbottom=True)
        ax_dict['ax_panel_bottom']=ax_panel_bottom
    
    if field_right_panel_dict is not None:
        ax_panel_right:plt.Axes=ax_dict['ax_panel_right']
        ax_panel_right= plot_multiple_1D_fields(
            coordinate_direction='horizontal',
            coordinate=y_coordinate,
            ax=ax_panel_right,
            field_dict_list=field_right_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_right_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_right_panel_dict.get('axvline_dict_list',None),
            xmin=field_right_panel_dict.get('vmin',vmin),xmax=field_right_panel_dict.get('vmax',vmax),
            ymin=ymin,ymax=ymax,
            xlabel=field_right_panel_dict.get('label',label),ylabel=ylabel,
            xscale=field_right_panel_dict.get('scale','linear'),yscale='linear',
            plot_legend=field_right_panel_dict.get('plot_legend',False),
            name=field_right_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        ax_panel_right.tick_params(axis='y', labelrotation=0,left=False,right=True,labelleft=False,labelright=True)
        ax_panel_right.tick_params(axis='x', labelrotation=0,top=True,bottom=True,labeltop=True,labelbottom=True)
        ax_dict['ax_panel_right']=ax_panel_right
    if field_left_panel_dict is not None:
        ax_panel_left:plt.Axes=ax_dict['ax_panel_left']
        ax_panel_left= plot_multiple_1D_fields(
            coordinate_direction='horizontal',
            coordinate=y_coordinate,
            ax=ax_panel_left,
            field_dict_list=field_left_panel_dict.get('field_dict_list',None),
            axhline_dict_list=field_left_panel_dict.get('axhline_dict_list',None),
            axvline_dict_list=field_left_panel_dict.get('axvline_dict_list',None),
            xmin=field_left_panel_dict.get('vmin',vmin),xmax=field_left_panel_dict.get('vmax',vmax),
            ymin=ymin,ymax=ymax,
            xlabel=field_left_panel_dict.get('label',label),ylabel=ylabel,
            xscale=field_left_panel_dict.get('scale','linear'),yscale='linear',
            plot_legend=field_left_panel_dict.get('plot_legend',False),
            name=field_left_panel_dict.get('title',None),
            return_fig=True,
        )['ax_main']
        ax_panel_left.tick_params(axis='y', labelrotation=0,left=True,right=False,labelleft=True,labelright=False)
        ax_panel_left.tick_params(axis='x', labelrotation=0,top=True,bottom=True,labeltop=True,labelbottom=True)
        ax_dict['ax_panel_left']=ax_panel_left
    ax_main.tick_params(axis='x', labelrotation=0,top=field_top_panel_dict is None,bottom=field_bottom_panel_dict is None,labeltop=field_top_panel_dict is None,labelbottom=field_bottom_panel_dict is None)
    ax_main.tick_params(axis='y', labelrotation=0,left=field_left_panel_dict is None,right=field_right_panel_dict is None,labelleft=field_left_panel_dict is None,labelright=field_right_panel_dict is None)
    if plot_colorbar:
        if ax_cbar is None:
            ax_cbar:plt.Axes=ax_dict.get('ax_cbar_0',ax_dict['ax_main'])
        fig.colorbar(pcm, cax=ax_cbar)
        ax_cbar.set_ylabel(label,fontsize=15)
        ax_dict['ax_cbar']=ax_cbar
    fig.suptitle(name,fontsize=15)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=ax_dict['fig'],fig_path=os.path.join(working_dir,'%s.png' %(name)))

def plot_2D_field(
    field:np.ndarray,
    ax_dict:Optional[dict]=None,ax_cbar:Optional[plt.Axes]=None,
    x_coordinate=[0],y_coordinate=[0],
    threshold:Optional[float]=None,vmin:Optional[float]=-1,vmax:Optional[float]=1,norm:Optional[Normalize]=None,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    cmap='seismic',alpha:float=1.0,aspect=1.0,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    zoom:float=1.0,
    plot_profile=True,profile_at_x:Optional[int|list[int]]=None,profile_at_y:Optional[int|list[int]]=None,
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.'):
    """
    All variables should be normalized in advance
        ax_dict:
        {
            'fig':fig,
            'ax_main':ax_main,   #must be provided
            'ax_panel_top':ax_panel_top,   #can be None
            'ax_panel_right':ax_panel_right,   #can be None
            'ax_panel_bottom':ax_panel_bottom,   #can be None
            'ax_panel_left':ax_panel_left,   #can be None
            'ax_cbar':ax_cbar,
        }
        profile_at_x: float or list[float], optional
            Single list of x positions for vertical profiles (along y-direction)
        profile_at_y: float or list[float], optional
            Single list of y positions for horizontal profiles (along x-direction)
    """
    field=np.asarray(field)
    x_coordinate=np.asarray(x_coordinate).flatten()
    y_coordinate=np.asarray(y_coordinate).flatten()
    n_x=x_coordinate.size
    n_y=y_coordinate.size
    assert field.shape==(n_x,n_y)
    if threshold!=None:
        field_masked = np.where(np.abs(field) >= threshold, field, np.nan)
    else:
        field_masked=field
    if xmin is None:
        xmin=np.nanmin(x_coordinate)
    if xmax is None:
        xmax=np.nanmax(x_coordinate)
    if ymin is None:
        ymin=np.nanmin(y_coordinate)
    if ymax is None:
        ymax=np.nanmax(y_coordinate)
    norm=Normalize(vmin=vmin,vmax=vmax,clip=True)
    field_max_id=tuple(np.asarray(np.where(np.abs(field)==np.nanmax(np.abs(field))),dtype=np.int32)[:,0])   #field_max_id=(x_id,y_id)
    
    if plot_profile:
        
        # Convert profile_at_x to array
        if profile_at_x is None:
            profile_at_x_id_list=[field_max_id[0]]
        else:
            profile_at_x_id_list=get_closest_coordinate_id(coordinate=x_coordinate,pos=profile_at_x)
        
        # Convert profile_at_y to array
        if profile_at_y is None:
            profile_at_y_id_list=[field_max_id[1]]
        else:
            profile_at_y_id_list=get_closest_coordinate_id(coordinate=y_coordinate,pos=profile_at_y)
        
        # Create colormap and normalizers for colors
        # Normalize based on index values for color mapping
        x_color_norm = Normalize(vmin=np.min(profile_at_x_id_list), vmax=np.max(profile_at_x_id_list))
        y_color_norm = Normalize(vmin=np.min(profile_at_y_id_list), vmax=np.max(profile_at_y_id_list))
        x_profile_color=plt.get_cmap('tab10_r')
        y_profile_color=plt.get_cmap('tab20')
        
        # Build field_dict_list for multiple vertical profiles (top panel)
        field_dict_list_left = []
        field_dict_list_top = []
        axvline_dict_list = []
        axhline_dict_list = []
        for i, x_id in enumerate(profile_at_x_id_list):
            color = x_profile_color(x_color_norm(x_id))
            field_dict_list_left.append({
                'field': field[x_id, :],
                'label': f'{xlabel}={x_coordinate[x_id]:.2f}',
                'color': color,
            })
            axvline_dict_list.append({
                'x': x_coordinate[x_id],
                'color': color,
            })
        field_top_panel_dict={
            'field_dict_list':field_dict_list_top,
            'axvline_dict_list':axvline_dict_list,
            'plot_legend':True,
        }
        for i,y_id in enumerate(profile_at_y_id_list):
            color = y_profile_color(y_color_norm(y_id))
            field_dict_list_top.append({
                'field': field[:, y_id],
                'label': f'{ylabel}={y_coordinate[y_id]:.2f}',
                'color': color,
            })
            axhline_dict_list.append({
                'y': y_coordinate[y_id],
                'color': color,
            })
        field_left_panel_dict={
            'field_dict_list':field_dict_list_left,
            'axhline_dict_list':axhline_dict_list,
            'plot_legend':True,
        }
    else:
        field_top_panel_dict=None
        field_left_panel_dict=None

    ax_dict=plot_2D_field_with_side_panel(
        field=field_masked,field_top_panel_dict=field_top_panel_dict,field_left_panel_dict=field_left_panel_dict,
        ax_dict=ax_dict,ax_cbar=ax_cbar,
        x_coordinate=x_coordinate,y_coordinate=y_coordinate,
        vmin=vmin,vmax=vmax,cmap=cmap,norm=norm,alpha=alpha,
        xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
        label=label,xlabel=xlabel,ylabel=ylabel,
        zoom=zoom,plot_colorbar=plot_colorbar,aspect=aspect,
        return_fig=True,name=name,working_dir=working_dir,
    )
    ax_main: plt.Axes = ax_dict['ax_main']
    if plot_profile:
        plot_multiple_1D_fields(
            coordinate=x_coordinate,
            axhline_dict_list=axhline_dict_list,
            axvline_dict_list=axvline_dict_list,
            plot_legend=False,
            ax=ax_main,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,xlabel=xlabel,ylabel=ylabel,
        )
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=ax_dict['fig'],fig_path=os.path.join(working_dir,'%s.png' %(name)))


def plot_multiple_2D_fields(
    field_dict_list:list[dict],x_coordinate,y_coordinate,
    xmin:Optional[float]=None,xmax:Optional[float]=None,ymin:Optional[float]=None,ymax:Optional[float]=None,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',xlabel=r'$\frac{x}{\lambda_0}$',ylabel=r'$\frac{y}{\lambda_0}$',
    zoom:float=1.0,
    return_fig=True,name='',working_dir='.',
    aspect=1.0,
    ):
    """
    Args:
        field_dict_list=
            [
                {
                    'field':field,
                    'vmin':None,'vmax':None,'cmap':'seismic',
                    'label':label,
                    'threshold':None,
                    'plot_profile':False,'profile_at_x':Optional[int]=None,'profile_at_y':Optional[int]=None,
                    'plot_colorbar':True,
                    'alpha':1.0,
                },
            ...
            ] 
        x_coordinate (_type_): _description_
        y_coordinate (_type_): _description_
        xmin (Optional[float], optional): _description_. Defaults to None.
        xmax (Optional[float], optional): _description_. Defaults to None.
        ymin (Optional[float], optional): _description_. Defaults to None.
        ymax (Optional[float], optional): _description_. Defaults to None.
        label (regexp, optional): _description_. Defaults to ''.
        xlabel (regexp, optional): _description_. Defaults to ''.
        ylabel (regexp, optional): _description_. Defaults to ''.
        zoom (float, optional): _description_. Defaults to 1.0.
        name (str, optional): _description_. Defaults to ''.
    """
    plot_profile=any([field_dict.get('plot_profile',False) for field_dict in field_dict_list])
    field_num=len(field_dict_list)
    l_x=xmax-xmin
    l_y=ymax-ymin
    ax_main_height=6   #unit: inch
    ax_main_width=ax_main_height*l_x/l_y   #unit: inch
    ax_dict=generate_side_panel_axes(
        ax_main_height=ax_main_height,ax_main_width=ax_main_width,
        generate_ax_panel_top=plot_profile,
        generate_ax_panel_left=plot_profile,
        ax_cbar_num=field_num,
    )
    for i, field_dict in enumerate(field_dict_list):
        ax_dict=plot_2D_field(
            field=field_dict['field'],
            ax_dict=ax_dict,ax_cbar=ax_dict[f'ax_cbar_{i}'],
            x_coordinate=x_coordinate,y_coordinate=y_coordinate,
            threshold=field_dict.get('threshold',None),vmin=field_dict.get('vmin',None),vmax=field_dict.get('vmax',None),
            xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax,
            cmap=field_dict.get('cmap','seismic'),alpha=field_dict.get('alpha',1.0),aspect=aspect,
            label=field_dict.get('label',label),xlabel=xlabel,ylabel=ylabel,
            return_fig=True,
            plot_profile=field_dict.get('plot_profile',False),
            profile_at_x=field_dict.get('profile_at_x',None),
            profile_at_y=field_dict.get('profile_at_y',None),
            zoom=zoom,plot_colorbar=field_dict.get('plot_colorbar',True),
        )
    fig:plt.Figure=ax_dict['fig']
    fig.suptitle(name,fontsize=15)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=fig,fig_path=os.path.join(working_dir,'%s.png' %(name)))


def plot_polar_field(
    field:np.ndarray,
    r_coordinate:np.ndarray,a_coordinate:np.ndarray,
    ax_dict:Optional[dict]=None,
    r_min:Optional[float]=None,r_max:Optional[float]=None,
    a_min:Optional[float]=None,a_max:Optional[float]=None,
    v_min:Optional[float]=-1,v_max:Optional[float]=1,norm:Optional[Normalize]=None,
    cmap='seismic',alpha=1.0,
    label=r'$a=\frac{E}{E_c}=\frac{B}{B_c}$',r_label='',a_label='θ',
    plot_colorbar=True,
    return_fig=True,name='',working_dir='.',
):
    """
    All variables should be normalized in advance
    """
    field=np.asarray(field)
    a_coordinate=np.array(a_coordinate).flatten()
    r_coordinate=np.array(r_coordinate).flatten()
    N_a=a_coordinate.size
    N_r=r_coordinate.size
    assert field.shape==(N_a,N_r)
    if r_min is None:
        r_min=0
    if r_max is None:
        r_max=np.nanmax(r_coordinate)
    if a_min is None:
        a_min=-np.pi
    if a_max is None:
        a_max=np.pi
    if v_min is None:
        v_min=np.nanmin(field)
    if v_max is None:
        v_max=np.nanmax(field)
    if norm is None:
        norm=Normalize(vmin=v_min,vmax=v_max,clip=True)
    if ax_dict is None:
        fig = plt.figure(figsize=(6, 6))
        ax_main=fig.add_subplot(111, projection='polar')
        ax_dict={'fig':fig,'ax_main':ax_main}
    fig:plt.Figure=ax_dict['fig']
    ax_main:plt.Axes=ax_dict['ax_main']
    pcm=ax_main.pcolormesh(a_coordinate, r_coordinate, field.T, cmap=cmap, shading='auto', norm=norm, alpha=alpha)
    ax_main.set_xlabel(a_label, fontsize=15)
    ax_main.set_ylabel(r_label, fontsize=15, labelpad=20)
    ax_main.set_thetamin(np.degrees(a_min))
    ax_main.set_thetamax(np.degrees(a_max))
    ax_main.set_rlim(r_min, r_max)
    ax_main.set_theta_direction(1)
    ax_main.set_theta_zero_location('E')
    if plot_colorbar:
        ax_cbar=fig.colorbar(pcm, ax=ax_main)
        ax_cbar.set_label(label, fontsize=15)
        ax_dict['ax_cbar']=ax_cbar
    ax_main.grid(True, linestyle='--', alpha=0.7)
    fig.suptitle(name, fontsize=15)
    if return_fig:
        return ax_dict
    else:
        return savefig(fig=fig, fig_path=os.path.join(working_dir, f'{name}.png'))


def plot_quiver_field(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
    Vx: np.ndarray,
    Vy: np.ndarray,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,ax_cbar: Optional[plt.Axes] = None,
    step_x: int = 1,step_y: int = 1,
    aspect: Optional[float] = 1,
    threshold: float = 0.0,
    Bmin: Optional[float] = None,
    Bmax: Optional[float] = None,
    xmin: Optional[float] = None,xmax: Optional[float] = None,ymin: Optional[float] = None,ymax: Optional[float] = None,
    cmap: str = 'viridis',
    xlabel: str = 'x',
    ylabel: str = 'y',
    label: str = 'Magnitude',
    scale: Optional[float] = None,
    return_fig=True,name='',working_dir='.',
) -> None:
    """
    Plot a 2D vector field with arrow lengths and colors controlled by optional scalar fields.

    Parameters
    ----------
    x_coordinate : 1D array, shape (Nx,)
        x-coordinates.
    y_coordinate : 1D array, shape (Ny,)
        y-coordinates.
    Vx : 2D array, shape (Nx, Ny)
        x-component of the vector field.
    Vy : 2D array, shape (Nx, Ny)
        y-component of the vector field.
    A : 2D array, shape (Nx, Ny), optional
        Scalar field determining arrow lengths. Should be non-negative. If None, the vector magnitude is used.
    B : 2D array, shape (Nx, Ny), optional
        Scalar field determining arrow colors. If None, the vector magnitude is used.
    threshold : float, optional
        Minimum vector magnitude to plot an arrow. Vectors with magnitude below this threshold are ignored.
    step_x : int, optional
        Plotting step in the x-direction: only every `step_x`-th point is drawn.
    step_y : int, optional
        Plotting step in the y-direction: only every `step_y`-th point is drawn.
    aspect : float, optional
        Aspect ratio of the plot. If None, aspect=1 is used.
    cmap : str, optional
        Colormap name for the arrow colors.
    name : str, optional
        Plot title.
    xlabel, ylabel : str, optional
        Axis labels.
    label : str, optional
        Label for the colorbar.
    """
    # Check input shapes
    x_coordinate = np.asarray(x_coordinate).flatten()
    y_coordinate = np.asarray(y_coordinate).flatten()
    Nx=x_coordinate.size
    Ny=y_coordinate.size
    # Compute vector magnitude as default for A and/or B
    magnitude = np.linalg.norm([Vx,Vy],axis=0)
    
    if A is None:
        A = magnitude
    else:
        A = np.asarray(A)
    assert np.all(np.nan_to_num(A) >= 0), "A must be non-negative to represent arrow lengths"
    Amax=np.nanmax(A)
    if B is None:
        B = magnitude
    else:
        B = np.asarray(B)
    
    if Bmin is None:
        Bmin = np.nanmin(B)
    if Bmax is None:
        Bmax = np.nanmax(B)
    if scale is None:
        lx=xmax-xmin
        scale=100*Amax/lx if lx>0 else 1.0
    norm=Normalize(vmin=Bmin, vmax=Bmax)
    Vx=np.asarray(Vx)
    Vy=np.asarray(Vy)
    # Ensure all arrays have consistent shape
    assert A.shape == (Nx, Ny) and B.shape == (Nx, Ny) and Vx.shape == (Nx, Ny) and Vy.shape == (Nx, Ny), "A and B must have the same shape as Vx and Vy"

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_coordinate, y_coordinate, indexing='ij')  # shape (Nx, Ny)

    # Compute unit direction vectors (avoid division by zero)
    # Scale direction vectors by A to get the final arrow components
    mask = magnitude > threshold
    U_dir=np.where(mask,A * Vx/magnitude,np.nan)
    V_dir=np.where(mask,A * Vy/magnitude,np.nan)

    # Subset data for plotting
    X_plot = X[::step_x, ::step_y]
    Y_plot = Y[::step_x, ::step_y]
    U_plot = U_dir[::step_x, ::step_y]
    V_plot = V_dir[::step_x, ::step_y]
    B_plot = B[::step_x, ::step_y]

    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    if ax_cbar is None:
        ax_cbar = ax
    # Quiver plot with color mapping based on B
    q = ax.quiver(
        X_plot, Y_plot, U_plot, V_plot, B_plot,
        cmap=cmap,
        norm=norm,
        scale_units='x',
        angles='xy',
        width=0.005,
        headwidth=1,
        headlength=5,
        scale=scale,
    )

    # Set axis properties
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(name)
    ax.set_aspect(aspect)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add colorbar
    cbar = plt.colorbar(q, cax=ax_cbar, label=label)

    plt.tight_layout()
    if return_fig:
        return {'fig': ax.get_figure(), 'ax_main': ax, 'colorbar': cbar}
    else:
        return savefig(fig=ax.get_figure(), fig_path=os.path.join(working_dir, f'{name}.png'))


if __name__ == "__main__":
    pass
