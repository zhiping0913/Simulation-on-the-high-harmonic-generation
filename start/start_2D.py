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

def set_blocks(laser_a0=10,N=300,D=0.005,L=0,Kappa=0,theta_degree=0):
    # Input parameters
    #theta_degree = 45 		# Laser angle of incidence
    ND_a0=N*D/laser_a0
    theta_rad=np.radians(theta_degree)
    
    laser_lambda = 0.8*C.micron		# Laser wavelength, microns
    laser_omega0=(2*C.pi*C.speed_of_light)/(laser_lambda)
    laser_period=laser_lambda/C.speed_of_light
    #laser_a0 = 20		# Laser field strength
    laser_Bc=(C.m_e*laser_omega0)/(C.elementary_charge)
    laser_Ec=laser_Bc*C.speed_of_light
    #laser_amp=laser_a0*laser_Ec
    laser_intensity=(2*C.pi**2*C.speed_of_light**5*C.epsilon_0*C.m_e**2*laser_a0**2)/(C.elementary_charge**2*laser_lambda**2)
    laser_FWHM=8*C.femto   #The full width at half maximum of the intensity.
    laser_tau=laser_FWHM/math.sqrt(2*math.log(2))
    #laser_tau=laser_FWHM/1.66510
    laser_Nc=laser_omega0**2*C.m_e*C.epsilon_0/C.elementary_charge**2
    laser_phase = C.pi		# Laser carrier envelope phase at the peak of the envelope
    laser_delay=5*laser_period   #How long the peak will arrive at x_xim
    laser_w_0_lambda= 12.5   #Beam waist at focus (1/e radius), unit: laser_lambda
    laser_z_0_lambda= C.pi*laser_w_0_lambda**2   #Beam rayleigh range, unit: laser_lambda

    #N = 500		# Plasma density in critical densities
    target_Ne=N*laser_Nc
    #D = 0.005	# Target thickness, unit: lambda
    
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
    cells_per_lambda_x =1000 	# Number of cells per laser wavelength in x direction
    cells_per_lambda_y =500

    parts_nc =10 		# How many particles required to form one critical density
    FWHM_space  = 2 	# Scaling factor to determine space required for simulation
    vacuum_length_x_lambda=50   #lambda
    vacuum_length_y_lambda=50   #lambda
    target_width_lambda=vacuum_length_y_lambda*0.9*2   #The width of the tirget in y direction. Unit: lambda
    nparticles_per_cell=100
    nparticles=round(cells_per_lambda_x*cells_per_lambda_y*(D+L)*target_width_lambda*nparticles_per_cell)



    laser_w_x_min_lambda=laser_w_0_lambda*math.sqrt(1+(vacuum_length_x_lambda/laser_z_0_lambda)**2)   #spot size at x_min, unit: laser_lambda
    laser_radius_x_min_lambda=vacuum_length_x_lambda*(1+(laser_z_0_lambda/vacuum_length_x_lambda)**2)   #Radius of curvature on x_min
    laser_phase_x_min=laser_phase-(laser_delay/laser_period)*2*C.pi-math.atan(vacuum_length_x_lambda/laser_z_0_lambda)
    laser_intensity_x_min=laser_intensity*(laser_w_0_lambda/laser_w_x_min_lambda)   #note that it is I0*(w0/wx) not I0*(w0/wx)^2 in 2D simulation
    
    constant=block(built='constant')
    constant.title='0#2D simulation for 0.8μm laser interacting with overdense plasma'
    #constant.theta_rad=theta_rad
    constant.laser_lambda='%e #unit: m' %(laser_lambda)
    constant.laser_period='%e #unit: s' %(laser_period)
    constant.laser_a0=laser_a0
    constant.laser_FWHM='%e #unit: s' %(laser_FWHM)
    constant.laser_Nc='%e #unit: m^-3' %(laser_Nc)
    constant.Kappa=Kappa
    constant.N=N
    constant.D=D
    constant.L=L
    constant.target_Ne='N*laser_Nc #unit: m^-3'
    constant.target_curvature='Kappa/laser_lambda #unit: m^-1'
    constant.target_thickness='D*laser_lambda #unit: m'
    constant.target_Ne_gradient='L*laser_lambda #unit: m'
    constant.target_width='%e*laser_lambda #unit: m' %(target_width_lambda)
    constant.cells_per_lambda_x=cells_per_lambda_x
    constant.cells_per_lambda_y=cells_per_lambda_y
    constant.nparticles_per_cell=nparticles_per_cell
    constant.vacuum_length_x='%e*laser_lambda #unit: m' %(vacuum_length_x_lambda)
    constant.vacuum_length_y='%e*laser_lambda #unit: m' %(vacuum_length_y_lambda)
    constant.laser_w_0='%e*laser_lambda #unit: m' %(laser_w_0_lambda)
    constant.ion_mass=ion_mass
    constant.ion_charge=ion_charge
    if Kappa>0.0001:
        constant.R='1/Kappa #unit: lambda'
        constant.target_radius='R*laser_lambda #unit: m'
        constant.target_f='sqrt((x-target_radius)^2+y^2)-abs(target_radius) #unit: m'
    else:
        constant.target_f='x #unit: m'
    
    control=block(built='control')
    control.nx=round(2*vacuum_length_x_lambda*cells_per_lambda_x)
    control.ny=round(2*vacuum_length_y_lambda*cells_per_lambda_y)
    control.x_min='-vacuum_length_x'
    control.x_max='+vacuum_length_x'
    control.y_min='-vacuum_length_y'
    control.y_max='+vacuum_length_y'
    control.t_end=1.5*vacuum_length_x_lambda*laser_period
    control.stdout_frequency = 10
    control.use_multiphoton = 'F' 
    control.use_bsi = 'F'
    control.field_ionisation='F'
    control.physics_table_location='/home/zl8336/Software/Epoch/epoch-4.19.4/epoch2d/src/physics_packages/TABLES'


    boundaries=block(built='boundaries')
    boundaries.bc_x_min='simple_outflow'
    boundaries.bc_x_max='simple_outflow'
    boundaries.bc_y_min='simple_outflow'
    boundaries.bc_y_max='simple_outflow'

    laser=block(built='laser')
    laser.boundary = 'x_min'
    laser.intensity=laser_intensity_x_min
    laser.omega=laser_omega0
    laser.phase ='%e+%e*y^2' %(laser_phase_x_min, C.pi/(laser_radius_x_min_lambda*laser_lambda**2))
    laser.profile='gauss(sqrt(y^2),0,%e)' %(laser_w_x_min_lambda*laser_lambda)
    laser.t_profile = 'gauss(time, %e-%e*y^2, %e)' %(laser_delay,laser_period/(2*laser_radius_x_min_lambda*laser_lambda**2),laser_tau)
    laser.t_start = 0

    fields=block(built='fields')
    fields.ex="'%s'" %(os.path.join('../Initialize_Field','Electric_Field_Ex.dat'))
    fields.ey="'%s'" %(os.path.join('../Initialize_Field','Electric_Field_Ey.dat'))
    #fields.ez="'%s'" %(os.path.join('../Initialize_Field','Ez'))
    #fields.bx="'%s'" %(os.path.join('../Initialize_Field','Bx'))
    #fields.by="'%s'" %(os.path.join('../Initialize_Field','By'))
    fields.bz="'%s'" %(os.path.join('../Initialize_Field','Magnetic_Field_Bz.dat'))

    Electron=block(built='species')
    Electron.name = 'Electron'
    Electron.charge = -1.0
    Electron.mass = 1.0
    Electron.nparticles=nparticles
    #Electron.number_density="'%s'" %(os.path.join('../Initialize_Target','K_%+5.3f,NDa0_%3.2f' %(Kappa,ND_a0),'Ne.dat'))
    if L>0.0001:
        Electron.number_density ='if(abs(y) lt target_width/2,if(target_f lt -target_thickness/2,target_Ne*exp((target_f+target_thickness/2)/target_Ne_gradient),if(target_f lt target_thickness/2,target_Ne,0)),0)'
    else:
        Electron.number_density ='if(abs(y) lt target_width/2,if((target_f gt -target_thickness/2) and (target_f lt target_thickness/2),target_Ne,0),0)'
    
    Ion=block(built='species')
    Ion.name = 'Ion'
    Ion.charge = ion_charge
    Ion.mass = ion_mass
    Ion.nparticles=round(nparticles/ion_charge)
    #Ion.number_density="'%s'" %(os.path.join('../Initialize_Target','K_%+5.3f,NDa0_%3.2f' %(Kappa,ND_a0),'Ni.dat'))
    Ion.number_density ='density(Electron)/ion_charge'

    Silicon=block(built='species')
    Silicon.name='Silicon'
    Silicon.charge = 0 
    Silicon.mass=ion_mass
    Silicon.atomic_no = ion_number
    Silicon.ionise = 'T' 
    Silicon.unique_electron_species='T'
    Silicon.nparticles=round(nparticles/ion_charge)
    if L>0:
        #Silicon.number_density ='if(x lt -target_thickness/2,%e*exp((x+target_thickness/2)/target_Ne_gradient),if(x lt target_thickness/2,%e,0))' %(ion_density_M,ion_density_M)
        Silicon.number_density ='if(x lt 0,%e*exp(x/target_Ne_gradient),if(x lt target_thickness,%e,0))' %(ion_density,ion_density)
    else:
        Silicon.number_density ='if((x gt -target_thickness/2) and (x lt target_thickness/2),%e,0)' %(ion_density)
    Silicon.drift_y=-ion_mass*C.m_e*C.speed_of_light*math.tan(theta_rad)

    output_fields=block(built='output')
    output_fields.name = 'fields'
    output_fields.file_prefix='fields'
    #output_fields.dt_snapshot = 1.3*vacuum_length_x_lambda*laser_period
    output_fields.dt_snapshot = 10*laser_period
    output_fields.grid = 'always'
    output_fields.ex = 'always'
    output_fields.ey = 'always'
    #output_fields.ez = 'always'
    #output_fields.bx = 'always'
    #output_fields.by = 'always'
    output_fields.bz = 'always'
    output_fields.number_density = 'always + species + no_sum'
    #output_fields.particle_grid = 'always'
    #output_fields.vy = 'always'
    #output_fields.vx = 'always'
    #output_fields.id = 'always'
    #output_fields.gamma = 'always'
    
    output_restart=block(built='output')
    output_restart.name = 'restart'
    output_restart.file_prefix='restart'
    output_restart.dt_snapshot = 10*laser_period
    output_restart.restartable = 'T'
    
    block_list=[
        constant,
        control,
        boundaries,
        #laser,
        fields,
        Electron,
        Ion,
        #Silicon,
        output_fields,
        #output_restart,
        ]
    return block_list




def sbatch(working_dir:str, block_list:list[block]):
    epoch_path='/home/zl8336/Software/Epoch/epoch-4.19.5/epoch2d/bin/epoch2d'
    slurm="""#!/bin/bash
#SBATCH --account=mikhailova
#SBATCH --job-name=test   # create a name for your job
#SBATCH --nodes=4                # node count
#SBATCH --ntasks=448               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem=900G         # memory
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


N=350
a0=20
theta_degree=45

ND_a0=0.3
L=0
D=ND_a0*a0/N-L

Kappa=-0.005

working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface','a0=%d'%(a0),'2D','K=%+5.3f,D=%3.2f,L=%3.2f' %(Kappa,D,L))
print(working_dir)
block_list=set_blocks(laser_a0=a0,N=N,D=D,Kappa=Kappa,L=L,theta_degree=theta_degree)
sbatch(working_dir,block_list)
exit(0)



for ND_a0 in [0.3,0.5,1,50]:
    for Kappa in [-0.01,-0.008,-0.005,-0.002,0.0,0.005]:
        for L in [0.0,0.05]:
            D=ND_a0*a0/N-L
            if D<0:
                continue
            else:
                working_dir=os.path.join('/scratch/gpfs/MIKHAILOVA/zl8336/Curved_surface','a0=%d'%(a0),'2D','K=%+5.3f,D=%3.2f,L=%3.2f' %(Kappa,D,L))
                print(working_dir)

                block_list=set_blocks(laser_a0=a0,N=N,D=D,Kappa=Kappa,L=L,theta_degree=theta_degree)
                sbatch(working_dir,block_list)


exit(0)













