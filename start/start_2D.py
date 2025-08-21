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

def set_blocks(laser_a0=10,target_Ne_nc=300,D_lambda=0.005,L_lambda=0,a_lambda=0,theta_degree=0):
    # Input parameters
    #theta_degree = 45 		# Laser angle of incidence
    theta_rad=np.radians(theta_degree)
    
    laser_lambda = 0.8*C.micron		# Laser wavelength, microns
    laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_period=laser_lambda/C.speed_of_light
    #laser_a0 = 20		# Laser field strength
    laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
    laser_intensity=(2*C.pi**2*C.speed_of_light**5*C.epsilon_0*C.m_e**2*laser_a0**2)/(C.elementary_charge**2*laser_lambda**2)
    laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
    laser_duration=laser_FWHM/math.sqrt(2*math.log(2))
    #laser_duration=laser_FWHM/1.66510
    laser_Nc=laser_omega**2*C.m_e*C.epsilon_0/C.elementary_charge**2
    laser_phase = C.pi		# Laser carrier envelope phase at the peak of the envelope
    laser_delay=5*laser_period   #How long the peak will arrive at x_xim
    laser_w_0_lambda= 3   #Beam waist at focus (1/e radius), unit: laser_lambda
    laser_z_0_lambda= C.pi*laser_w_0_lambda**2   #Beam rayleigh range, unit: laser_lambda

    #target_Ne_nc = 500		# Plasma density in critical densities
    target_Ne=target_Ne_nc*laser_Nc
    #D_lambda = 0.005	# Target thickness, unit: lambda
    
    target_temperature_electron=0
    target_temperature_proton=0

    input_gradOlambda = 0.0	# Plasma density gradient scale length, wavelengths
    input_tempe = 0.0		# Initial electron temperature, eV
    input_tempion = 0.0		# Initial ion temperature, eV
    ion_atomic_weight=12
    ion_mass = 1836.1*ion_atomic_weight	# Ion to electron mass ratio
    ion_charge=6
    ion_number=14
    ion_mass_density=2.329085*1e3   #kg/m^3
    ion_molar_mass=ion_atomic_weight/1e3   #kg/mol
    ion_density=C.N_A*ion_mass_density/ion_molar_mass   #1/m^3
    input_polarization = 0 	# 0 for P, 1 for S	
    # Simulation control parameters
    cells_per_lambda_x =200 	# Number of cells per laser wavelength in x direction
    cells_per_lambda_y =200

    parts_nc =10 		# How many particles required to form one critical density
    FWHM_space  = 2 	# Scaling factor to determine space required for simulation
    vacuum_length_x_lambda=20   #lambda
    vacuum_length_y_lambda=10   #lambda
    target_width_lambda=vacuum_length_y_lambda*0.6*2   #The width of the tirget in y direction. Unit: lambda
    nparticles_per_cell=100
    nparticles=round(cells_per_lambda_x*cells_per_lambda_y*(D_lambda+L_lambda)*target_width_lambda*nparticles_per_cell)



    laser_w_x_min_lambda=laser_w_0_lambda*math.sqrt(1+(vacuum_length_x_lambda/laser_z_0_lambda)**2)   #spot size at x_min, unit: laser_lambda
    laser_radius_x_min_lambda=vacuum_length_x_lambda*(1+(laser_z_0_lambda/vacuum_length_x_lambda)**2)   #Radius of curvature on x_min
    laser_phase_x_min=laser_phase-(laser_delay/laser_period)*2*C.pi-math.atan(vacuum_length_x_lambda/laser_z_0_lambda)
    laser_intensity_x_min=laser_intensity*(laser_w_0_lambda/laser_w_x_min_lambda)   #note that it is I0*(w0/wx) not I0*(w0/wx)^2 in 2D simulation
    
    constant=block(built='constant')
    constant.title='0#2D simulation for 0.8μm laser interacting with overdense plasma'
    constant.theta_rad=theta_rad
    constant.laser_lambda=laser_lambda
    constant.laser_a0=laser_a0
    constant.laser_FWHM=laser_FWHM
    constant.laser_Nc=laser_Nc
    constant.a='%e/laser_lambda' %(a_lambda)
    constant.N=target_Ne_nc
    constant.D=D_lambda
    constant.L=L_lambda
    constant.target_Ne='N*laser_Nc'
    constant.target_thickness='D*laser_lambda'
    constant.target_Ne_gradient='L*laser_lambda'
    constant.target_width='%e*laser_lambda' %(target_width_lambda)
    constant.cells_per_lambda_x=cells_per_lambda_x
    constant.cells_per_lambda_y=cells_per_lambda_y
    constant.nparticles_per_cell=nparticles_per_cell
    constant.vacuum_length_x='%e*laser_lambda' %(vacuum_length_x_lambda)
    constant.vacuum_length_y='%e*laser_lambda' %(vacuum_length_y_lambda)
    constant.laser_w_0='%e*laser_lambda' %(laser_w_0_lambda)
    constant.ion_mass=ion_mass
    constant.ion_charge=ion_charge
    constant.x_rot='x*cos(theta_rad) + y*sin(theta_rad)'
    constant.y_rot='-x*sin(theta_rad) + y*cos(theta_rad)'
    constant.parabola='x_rot-a*y_rot^2'
    
    control=block(built='control')
    control.nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
    control.ny=round(2*vacuum_length_y_lambda*cells_per_lambda_y)
    control.x_min=-vacuum_length_x_lambda*laser_lambda
    control.x_max=vacuum_length_x_lambda*laser_lambda
    control.y_min=-vacuum_length_y_lambda*laser_lambda
    control.y_max=vacuum_length_y_lambda*laser_lambda
    control.t_end=50*laser_period
    control.stdout_frequency = 10
    control.use_multiphoton = 'F' 
    control.use_bsi = 'F'
    control.field_ionisation='F'
    control.physics_table_location='/home/zl8336/Software/Epoch/epoch-4.19.4/epoch2d/src/physics_packages/TABLES'


    boundaries=block(built='boundaries')
    boundaries.bc_x_min='simple_laser'
    boundaries.bc_x_max='simple_outflow'
    boundaries.bc_y_min='simple_outflow'
    boundaries.bc_y_max='simple_outflow'

    laser=block(built='laser')
    laser.boundary = 'x_min'
    laser.intensity=laser_intensity_x_min
    laser.omega=laser_omega
    laser.phase ='%e+%e*y^2' %(laser_phase_x_min, C.pi/(laser_radius_x_min_lambda*laser_lambda**2))
    laser.profile='gauss(sqrt(y^2),0,%e)' %(laser_w_x_min_lambda*laser_lambda)
    laser.t_profile = 'gauss(time, %e-%e*y^2, %e)' %(laser_delay,laser_period/(2*laser_radius_x_min_lambda*laser_lambda**2),laser_duration)
    laser.t_start = 0

    fields=block(built='fields')
    fields.ex="'%s'" %(os.path.join(working_dir,'Ex'))
    fields.ey="'%s'" %(os.path.join(working_dir,'Ey'))
    fields.ez="'%s'" %(os.path.join(working_dir,'Ez'))
    fields.bx="'%s'" %(os.path.join(working_dir,'Bx'))
    fields.by="'%s'" %(os.path.join(working_dir,'By'))
    fields.bz="'%s'" %(os.path.join(working_dir,'Bz'))

    Electron=block(built='species')
    Electron.name = 'Electron'
    Electron.charge = -1.0
    Electron.mass = 1.0
    Electron.nparticles=nparticles
    if L_lambda>0:
        #Electron.number_density ='if(abs(y_rot) lt target_width/2,if(x_rot lt -target_thickness/2,target_Ne*exp((x_rot+target_thickness/2)/target_Ne_gradient),if(x_rot lt target_thickness/2,target_Ne,0)),0)'
        Electron.number_density ='if(abs(y_rot) lt target_width/2,if(x_rot lt 0,target_Ne*exp(x_rot/target_Ne_gradient),if(x_rot lt target_thickness,target_Ne,0)),0)'
    else:
        Electron.number_density ='if(abs(y_rot) lt target_width/2,if((x_rot gt -target_thickness/2) and (x_rot lt target_thickness/2),target_Ne,0),0)'
    Electron.number_density ='if((abs(y_rot) lt target_width/2) and ((parabola-15*laser_lambda) gt -target_thickness/2) and ((parabola-15*laser_lambda) lt target_thickness/2),target_Ne,0)'
    
    Ion=block(built='species')
    Ion.name = 'Ion'
    Ion.charge = ion_charge
    Ion.mass = ion_mass
    Ion.nparticles=round(nparticles/ion_charge)
    Ion.number_density='density(Electron)/ion_charge'

    Silicon=block(built='species')
    Silicon.name='Silicon'
    Silicon.charge = 0 
    Silicon.mass=ion_mass
    Silicon.atomic_no = ion_number
    Silicon.ionise = 'T' 
    Silicon.unique_electron_species='T'
    Silicon.nparticles=round(nparticles/ion_charge)
    if L_lambda>0:
        #Silicon.number_density ='if(x lt -target_thickness/2,%e*exp((x+target_thickness/2)/target_Ne_gradient),if(x lt target_thickness/2,%e,0))' %(ion_density_M,ion_density_M)
        Silicon.number_density ='if(x lt 0,%e*exp(x/target_Ne_gradient),if(x lt target_thickness,%e,0))' %(ion_density,ion_density)
    else:
        Silicon.number_density ='if((x gt -target_thickness/2) and (x lt target_thickness/2),%e,0)' %(ion_density)
    Silicon.drift_y=-ion_mass*C.m_e*C.speed_of_light*math.tan(theta_rad)

    output=block(built='output')
    output.name = 'fields'
    output.dt_snapshot = laser_period
    output.grid = 'always'
    output.ex = 'always'
    output.ey = 'always'
    #output.ez = 'always'
    #output.bx = 'always'
    #output.by = 'always'
    output.bz = 'always'
    output.number_density = 'always + species + no_sum'
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
        Electron,
        Ion,
        #Silicon,
        output,
        ]
    return block_list




def sbatch(working_dir:str, block_list:list[block]):
    epoch_path='/home/zl8336/Software/Epoch/epoch-4.19.5/epoch2d/bin/epoch2d'
    slurm="""#!/bin/bash
#SBATCH --account=mikhailova
#SBATCH --job-name=test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=112               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem=640G         # memory
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

working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface/test_02'
block_list=set_blocks(laser_a0=3,target_Ne_nc=350,D_lambda=1,L_lambda=0,a_lambda=1/40,theta_degree=0)
sbatch(working_dir, block_list)

