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


working_dir='/scratch/gpfs/MIKHAILOVA/zl8336/45thick/ionisation=T/1D'

def set_blocks(laser_a0=10,target_ne_nc=300,D_lambda=3,L_lambda=0,theta_degree=0):
    # Input parameters
    #theta_degree = 45 		# Laser angle of incidence
    theta_rad=np.radians(theta_degree)
    
    laser_lambda = 0.8*C.micron		# Laser wavelength
    laser_omega=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_period=laser_lambda/C.speed_of_light
    #laser_a0 = 20		# Laser field strength
    laser_amp=laser_a0*(C.m_e*laser_omega*C.speed_of_light)/(C.elementary_charge)
    laser_FWHM=25*C.femto
    laser_duration=laser_FWHM/math.sqrt(2*math.log(2))
    laser_Nc=laser_omega**2*C.m_e*C.epsilon_0/C.elementary_charge**2
    laser_phase = C.pi		# Laser carrier envelope phase
    laser_delay=4*laser_duration   #How long the peak will arrive at x_xim, unit: s

    #target_ne_nc = 500		# Plasma density in critical densities
    target_ne=target_ne_nc*laser_Nc
    #D_lambda = 3	# Target thickness, unit: lambda
    #L_lambda=0.1   #target_Ne_gradient, unit: lambda
    
    target_temperature_electron=0
    target_temperature_Ion=0
    ion_atomic_weight=28
    ion_mass = 1836.1*ion_atomic_weight	# Ion to electron mass ratio
    ion_charge=12
    ion_number=14
    ion_mass_density=2.329085*1e3   #kg/m^3
    ion_molar_mass=ion_atomic_weight/1e3   #kg/mol
    ion_density=C.N_A*ion_mass_density/ion_molar_mass   #1/m^3
    input_polarization = 0 	# 0 for P, 1 for S	
    # Simulation control parameters
    cells_per_lambda = round(4000) 	# Number of cells per laser wavelength
    parts_nc = 60 		# How many particles required to form one critical density
    nparticles_per_cell=200
    nparticles=cells_per_lambda*(D_lambda+L_lambda)*nparticles_per_cell



    vacuum_length_x_lambda=100   #lambda

    #Moving 
    laser_omega_M=laser_omega*math.cos(theta_rad)
    laser_lambda_M=laser_lambda/math.cos(theta_rad)
    laser_amp_M=laser_amp*math.cos(theta_rad)
    laser_duration_M=laser_duration/math.cos(theta_rad)
    laser_phase_M=laser_phase-(laser_delay/laser_period)*2*C.pi*math.cos(theta_rad)
    target_Ne_M=target_ne/math.cos(theta_rad)
    ion_density_M=ion_density/math.cos(theta_rad)
    nparticles_M=nparticles/math.cos(theta_rad)**2
    
    constant=block(built='constant')
    constant.title='0#1D simulation for 0.8μm laser interacting with overdense plasma'
    constant.theta_degree=theta_degree
    constant.laser_lambda=laser_lambda
    constant.laser_a0=laser_a0
    constant.laser_FWHM=laser_FWHM
    constant.laser_Nc=laser_Nc
    constant.N=target_ne_nc
    constant.D=D_lambda
    constant.L=L_lambda
    constant.target_Ne='N*laser_Nc'
    constant.target_thickness='D*laser_lambda'
    constant.target_Ne_gradient='L*laser_lambda'
    constant.cells_per_lambda=cells_per_lambda
    constant.nparticles_per_cell=nparticles_per_cell
    constant.vacuum_length_x='%e*laser_lambda' %(vacuum_length_x_lambda)
    constant.ion_mass=ion_mass
    constant.ion_charge=ion_charge

    
    
    
    control=block(built='control')
    control.nx=round(2*vacuum_length_x_lambda*cells_per_lambda)
    control.x_min =-vacuum_length_x_lambda*laser_lambda
    control.x_max =vacuum_length_x_lambda*laser_lambda
    control.t_end=1.5*vacuum_length_x_lambda*laser_period
    control.stdout_frequency = 10
    control.use_multiphoton = 'T' 
    control.use_bsi = 'T'
    control.field_ionisation='T'
    control.physics_table_location='/home/zl8336/Software/Epoch/epoch-4.19.4/epoch1d/src/physics_packages/TABLES'

    boundaries=block(built='boundaries')
    boundaries.bc_x_min='simple_laser'
    boundaries.bc_x_max='simple_outflow'

    laser=block(built='laser')
    laser.boundary = 'x_min'
    laser.amp=laser_amp_M
    laser.omega=laser_omega_M
    laser.phase = laser_phase_M
    laser.polarisation_angle=C.pi/6
    laser.t_profile = 'gauss(time, %e, %e  )' %(laser_delay,laser_duration_M)
    laser.t_start = 0.0

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
    Electron.nparticles=round(nparticles_M)
    if L_lambda>0:
        #Electron.number_density ='if(x lt -target_thickness/2,%e*exp((x+target_thickness/2)/target_Ne_gradient),if(x lt target_thickness/2,%e,0))' %(target_Ne_M,target_Ne_M)
        Electron.number_density ='if(x lt 0,%e*exp(x/target_Ne_gradient),if(x lt target_thickness,%e,0))' %(target_Ne_M,target_Ne_M)
    else:
        Electron.number_density ='if((x gt -target_thickness/2) and (x lt target_thickness/2),%e,0)' %(target_Ne_M)
    Electron.drift_y=-C.m_e*C.speed_of_light*math.tan(theta_rad)

    Ion=block(built='species')
    Ion.name = 'Ion'
    Ion.charge = ion_charge
    Ion.mass = ion_mass
    Ion.nparticles=round(nparticles_M/ion_charge)
    Ion.number_density='density(Electron)/ion_charge'
    Ion.drift_y=-ion_mass*C.m_e*C.speed_of_light*math.tan(theta_rad)



    Silicon=block(built='species')
    Silicon.name='Silicon'
    Silicon.charge = 0 
    Silicon.mass=ion_mass
    Silicon.atomic_no = ion_number
    Silicon.ionise = 'T' 
    Silicon.unique_electron_species='T'
    Silicon.nparticles=round(nparticles_M/ion_charge)
    if L_lambda>0:
        #Silicon.number_density ='if(x lt -target_thickness/2,%e*exp((x+target_thickness/2)/target_Ne_gradient),if(x lt target_thickness/2,%e,0))' %(ion_density_M,ion_density_M)
        Silicon.number_density ='if(x lt 0,%e*exp(x/target_Ne_gradient),if(x lt target_thickness,%e,0))' %(ion_density_M,ion_density_M)
    else:
        Silicon.number_density ='if((x gt -target_thickness/2) and (x lt target_thickness/2),%e,0)' %(ion_density_M)
    Silicon.drift_y=-ion_mass*C.m_e*C.speed_of_light*math.tan(theta_rad)



    output=block(built='output')
    output.name = 'fields'
    output.dt_snapshot = 1*laser_period
    #output.time_start=10*laser_period
    output.grid = 'always'
    #output.ex = 'always'
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
        #Electron,
        #Ion,
        Silicon,
        output]
    return block_list




def sbatch(working_dir:str, block_list:list[block]):
    epoch_path='/home/zl8336/Software/Epoch/epoch-4.19.5/epoch1d/bin/epoch1d'
    slurm="""#!/bin/bash
#SBATCH --account=mikhailova
#SBATCH --job-name=test   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=112               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem=256G         # memory
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
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


block_list=set_blocks(laser_a0=3,target_ne_nc=350,D_lambda=6,L_lambda=0.05,theta_degree=45)
sbatch(working_dir, block_list)


