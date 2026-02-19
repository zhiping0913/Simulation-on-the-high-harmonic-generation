import scipy.constants as C
import math
import os
import subprocess
import numpy as np

class block():
    def __init__(self,built:str):
        self.built=built
    
    def print(self):
        output='begin : %s\n' %(self.__dict__['built'])
        for key in self.__dict__.keys():
            if key=='built':
                continue
            elif key=='supplemental':
                output=output+'\t%s\n' %(self.__dict__['supplemental'])
            else:
                output=output+"\t%s=%s\n" %(key,self.__dict__[key])
        output=output+'end : %s\n\n' %(self.__dict__['built'])
        print(output)
        return output

def set_blocks(laser_a0=10,target_ne_nc=300,target_thickness_lambda=0.005,theta_degree=0):
    # Input parameters
    #theta_degree = 45 		# Laser angle of incidence
    theta_rad=np.radians(theta_degree)
    
    laser_lambda = 0.8*C.micron		# Laser wavelength, microns
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_period=laser_lambda/C.speed_of_light
    #laser_a0 = 20		# Laser field strength
    laser_Ec=(C.m_e*laser_omega0*C.speed_of_light)/(C.elementary_charge)
    laser_amp=laser_a0*laser_Ec
    laser_intensity=(2*C.pi**2*C.speed_of_light**5*C.epsilon_0*C.m_e**2*laser_a0**2)/(C.elementary_charge**2*laser_lambda**2)
    laser_FWHM=5*C.femto   #The full width at half maximum of the intensity.
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2))
    laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
    laser_phase = C.pi		# Laser carrier envelope phase at the peak of the envelope
    laser_delay=5*laser_period   #How long the peak will arrive at x_xim
    laser_w_0_lambda= 3   #Beam waist at focus (1/e radius), unit: laser_lambda
    laser_z_0_lambda= C.pi*laser_w_0_lambda**2   #Beam rayleigh range, unit: laser_lambda

    #target_ne_nc = 500		# Plasma density in critical densities
    target_ne=target_ne_nc*laser_Nc
    #target_thickness_lambda = 0.005	# Target thickness, unit: lambda
    
    target_temperature_electron=0
    target_temperature_proton=0

    input_gradOlambda = 0.0	# Plasma density gradient scale length, wavelengths
    input_tempe = 0.0		# Initial electron temperature, eV
    input_tempion = 0.0		# Initial ion temperature, eV
    ion_mass = 1836	# Ion to electron mass ratio
    input_polarization = 0 	# 0 for P, 1 for S	
    # Simulation control parameters
    cells_per_lambda = round(30) 	# Number of cells per laser wavelength
    parts_nc =10 		# How many particles required to form one critical density
    FWHM_space  = 2 	# Scaling factor to determine space required for simulation
    vacuum_length_x_lambda=10   #lambda
    vacuum_length_y_lambda=5   #lambda
    vacuum_length_z_lambda=5   #lambda
    target_width_lambda=vacuum_length_y_lambda*0.8*2   #The width of the tirget in y direction. Unit: lambda
    target_height_lambda=vacuum_length_z_lambda*0.8*2   #The height of the tirget in z direction. Unit: lambda
    nparticles_per_cell=3
    nparticles=round(cells_per_lambda**3*target_thickness_lambda*target_width_lambda*target_height_lambda*nparticles_per_cell)

    # Default sizing parameters (default of 1 generally appropriate)
    sparam_time = .8 	# Shorten or lengthen simulation time outside defaults
    out_freq = .1 	# Output frequency

    laser_w_x_min_lambda=laser_w_0_lambda*math.sqrt(1+(vacuum_length_x_lambda/laser_z_0_lambda)**2)   #spot size at x_min, unit: laser_lambda
    laser_radius_x_min_lambda=vacuum_length_x_lambda*(1+(laser_z_0_lambda/vacuum_length_x_lambda)**2)   #Radius of curvature on x_min
    laser_phase_x_min=laser_phase-(laser_delay/laser_period)*2*C.pi-math.atan(vacuum_length_x_lambda/laser_z_0_lambda)
    laser_intensity_x_min=laser_intensity*(laser_w_0_lambda/laser_w_x_min_lambda)**2
    
    constant=block(built='constant')
    constant.title='0#3D simulation for 0.8μm laser interacting with overdense plasma'
    constant.theta_rad=theta_rad
    constant.laser_lambda=laser_lambda
    constant.laser_a0=laser_a0
    constant.laser_FWHM=laser_FWHM
    constant.laser_Nc=laser_Nc
    constant.target_density='%e*laser_Nc' %(target_ne_nc)
    constant.target_thickness='%e*laser_lambda' %(target_thickness_lambda)
    constant.cells_per_lambda=cells_per_lambda
    constant.nparticles_per_cell=nparticles_per_cell
    constant.vacuum_length_x='%e*laser_lambda' %(vacuum_length_x_lambda)
    constant.vacuum_length_y='%e*laser_lambda' %(vacuum_length_y_lambda)
    constant.vacuum_length_z='%e*laser_lambda' %(vacuum_length_z_lambda)
    constant.laser_w_0='%e*laser_lambda' %(laser_w_0_lambda)
    constant.ion_mass=ion_mass
    constant.x_rot='x*cos(theta_rad) + y*sin(theta_rad)'
    constant.y_rot='-x*sin(theta_rad) + y*cos(theta_rad)'
    
    control=block(built='control')
    control.nx=round(2*vacuum_length_x_lambda*cells_per_lambda)
    control.ny=round(2*vacuum_length_y_lambda*cells_per_lambda)
    control.nz=round(2*vacuum_length_z_lambda*cells_per_lambda)
    control.x_min=-vacuum_length_x_lambda*laser_lambda
    control.x_max=vacuum_length_x_lambda*laser_lambda
    control.y_min=-vacuum_length_y_lambda*laser_lambda
    control.y_max=vacuum_length_y_lambda*laser_lambda
    control.z_min=-vacuum_length_z_lambda*laser_lambda
    control.z_max=vacuum_length_z_lambda*laser_lambda
    control.t_end=20*laser_period
    control.stdout_frequency = 10

    boundaries=block(built='boundaries')
    boundaries.bc_x_min='simple_laser'
    boundaries.bc_x_max='simple_outflow'
    boundaries.bc_y_min='simple_outflow'
    boundaries.bc_y_max='simple_outflow'
    boundaries.bc_z_min='simple_outflow'
    boundaries.bc_z_max='simple_outflow'

    laser=block(built='laser')
    laser.boundary = 'x_min'
    laser.intensity=laser_intensity_x_min
    laser.omega=laser_omega0
    laser.phase ='%e+%e*(y^2+z^2)' %(laser_phase_x_min, C.pi/(laser_radius_x_min_lambda*laser_lambda**2))
    laser.profile='gauss(sqrt(y^2+z^2),0,%e)' %(laser_w_x_min_lambda*laser_lambda)
    laser.t_profile = 'gauss(time, %e-%e*y^2, %e+%e*y^2)' %(laser_delay,laser_period/(2*laser_radius_x_min_lambda*laser_lambda**2),laser_tau,laser_tau/(laser_radius_x_min_lambda*laser_lambda)**2/2)
    laser.t_start = 0.0

    fields=block(built='fields')
    fields.ex="'%s'" %(os.path.join(working_dir,'E_x'))
    fields.ey="'%s'" %(os.path.join(working_dir,'E_y'))
    fields.ez="'%s'" %(os.path.join(working_dir,'E_z'))
    fields.bx="'%s'" %(os.path.join(working_dir,'B_x'))
    fields.by="'%s'" %(os.path.join(working_dir,'B_y'))
    fields.bz="'%s'" %(os.path.join(working_dir,'B_z'))
    
    
    Electron=block(built='species')
    Electron.name = 'Electron'
    Electron.charge = -1.0
    Electron.mass = 1.0
    Electron.nparticles=nparticles
    Electron.number_density ='if((x_rot gt %e) and (x_rot lt %e) and (y_rot gt %e) and (y_rot lt %e) and (z gt %e) and (z lt %e),%e,0)' %(-target_thickness_lambda*laser_lambda/2,target_thickness_lambda*laser_lambda/2,-target_width_lambda*laser_lambda/2,target_width_lambda*laser_lambda/2,-target_height_lambda*laser_lambda/2,target_height_lambda*laser_lambda/2,target_ne)

    Ion=block(built='species')
    Ion.name = 'Ion'
    Ion.charge = 1.0
    Ion.mass = ion_mass
    Ion.nparticles=nparticles
    Ion.number_density=Electron.number_density


    output=block(built='output')
    output.name = 'fields'
    output.dt_snapshot = laser_period
    output.grid = 'always'
    output.ex = 'always'
    output.ey = 'always'
    output.ez = 'always'
    #output.number_density = 'always + species + no_sum'
    #output.particle_grid = 'always'
    #output.vy = 'always'
    #output.vx = 'always'
    #output.id = 'always'
    #output.gamma = 'always'
    
    block_list=[
        constant,
        control,
        boundaries,
        #laser,
        fields,
        #Electron,
        #Ion,
        output
        ]
    return block_list




def sbatch(working_dir:str, block_list:list[block]):
    epoch_path='/home/zl8336/Software/Epoch/epoch-4.19.5/epoch3d/bin/epoch3d'
    slurm="""#!/bin/bash
#SBATCH --account=mikhailova
#SBATCH --job-name=test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=16               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem=128G         # memory
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --qos=standard
module purge
module load /opt/share/Modules/modulefiles/intel-rt/2024.2
module load /opt/share/Modules/modulefiles/intel-tbb/2021.13  
module load /opt/share/Modules/modulefiles/intel-oneapi/2024.2  
module load /opt/share/Modules/modulefiles/intel-mpi/oneapi/2021.13
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
srun --mpi=pmi2 %s < %s
""" %(epoch_path,os.path.join(working_dir,'deck.file'))
    os.makedirs(name=working_dir,exist_ok=True)
    os.chdir(path=working_dir)
    with open(file=os.path.join(working_dir,'input.deck'),mode='w') as input:
        for blocks in block_list:
            input.write(blocks.print())
    with open(file=os.path.join(working_dir,'deck.file'),mode='w') as deck:
        deck.write(working_dir)
    with open(file=os.path.join(working_dir,'slurm'),mode='w') as slurm_file:
        slurm_file.write(slurm)
    task=subprocess.Popen(args='sbatch %s' %(os.path.join(working_dir,'slurm')),shell=True,stdout=subprocess.PIPE)
    task.wait()
    print(task.communicate()[0].decode())

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Try_3D/dat'
block_list=set_blocks(laser_a0=10,target_ne_nc=450,target_thickness_lambda=100.0/450.0,theta_degree=0)
sbatch(working_dir, block_list)

