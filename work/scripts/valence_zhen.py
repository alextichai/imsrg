#!/usr/bin/env python

##########################################################################
##  goUniversal.py
##
##  A python script to run or submit jobs for the common use cases
##  of the IMSRG++ code. We check whether there is a pbs or slurm
##  scheduler, assign the relevant input parameters, set names
##  for the output files, and run or submit.
##  						-Ragnar Stroberg
##  						TRIUMF Nov 2016
######################################################################

from os import path,environ,mkdir,remove
from sys import argv
from subprocess import call,PIPE
from time import time,sleep
from datetime import datetime

### Check to see what type of batch submission system we're dealing with
BATCHSYS = 'NONE'
if call('type '+'qsub', shell=True, stdout=PIPE, stderr=PIPE) == 0: BATCHSYS = 'PBS'
elif call('type '+'srun', shell=True, stdout=PIPE, stderr=PIPE) == 0: BATCHSYS = 'SLURM'

### The code uses OpenMP and benefits from up to at least 24 threads
NTHREADS=64
exe = '%s/bin/imsrg++'%(environ['HOME'])

### Flag to swith between submitting to the scheduler or running in the current shell
#batch_mode=False
batch_mode=True
if 'terminal' in argv[1:]: batch_mode=False

### Don't forget to change this. I don't want emails about your calculations...
mail_address = 'andrea.porro@tu-darmstadt.de'

### This comes in handy if you want to loop over Z
ELEM = ['n','H','He','Li','Be','B','C','N',
       'O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K',
       'Ca','Sc','Ti','V','Cr','Mn','Fe','Co',  'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
       'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',  'Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
       'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb']# ,'Bi','Po','At','Rn','Fr','Ra','Ac','Th','U','Np','Pu']

### ARGS is a (string => string) dictionary of input variables that are passed to the main program
ARGS  =  {}

### Maximum value of s, and maximum step size ds
ARGS['smax'] = '500'
ARGS['dsmax'] = '0.5'

#ARGS['lmax3'] = '10' # for comparing with Heiko

### Norm of Omega at which we split off and start a new transformation
ARGS['omega_norm_max'] = '0.25'
#ARGS['hunter_gatherer'] = 'true'

### Model space parameters used for reading Darmstadt-style interaction files
ARGS['file2e1max'] = '14 file2e2max=28 file2lmax=-1'
#ARGS['file3e1max'] = '14 file3e2max=28 file3e3max=16'
ARGS['file3e1max'] = '18 file3e2max=36 file3e3max=24'
ARGS['hw'] = '16'
ARGS['2bme'] = '/home/li/src/NuHamil-public/exe/TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw16_emax14_e2max28.me2j.gz'
#ARGS['3bme'] = '/home/li/src/NuHamil-public/exe/EM1.8_2.0_NO2B_hw16_02/NO2B_ThBME_EM1.8_2.0_3NFJmax15_IS_hw16_ms14_28_16.stream.bin'
ARGS['3bme'] = '/data_share13/takayuki/me3j/NO2B_ThBME_EM1.8_2.0_3NFJmax15_JJmax15_IS_hw16_ms18_36_24.stream.bin'
ARGS['3bme_type'] = 'no2b'
ARGS['LECs'] = 'EM1.8_2.0'
### Name of a directory to write Omega operators so they don't need to be stored in memory. If not given, they'll just be stored in memory.
ARGS['scratch'] = '/data_share10/zhen/temp' # '/data_share10/zhen/temp'    

### Generator for core decoupling, can be atan, white, imaginary-time.  (atan is default)
ARGS['core_generator'] = 'atan' 
### Generator for valence deoupling, can be shell-model, shell-model-atan, shell-model-npnh, shell-model-imaginary-time (shell-model-atan is default)
ARGS['valence_generator'] = 'shell-model-atan' 

### Solution method: magnus, brueckner, flow, HF, MP3
ARGS['method'] = 'magnus'

### Tolerance for ODE solver if using flow solution method
ARGS['ode_tolerance'] = '1e-6'
ARGS['eta_criterion'] = '1e-5'

if BATCHSYS == 'PBS':
  FILECONTENT = """#!/bin/bash
#PBS -N %s
#PBS -q batchmpi
#PBS -d %s
#PBS -l walltime=192:00:00
#PBS -l nodes=1:ppn=%d
#PBS -l vmem=60gb
#PBS -m ae
#PBS -M %s
#PBS -j oe
#PBS -o imsrg_log/%s.o
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=%d
%s
  """

elif BATCHSYS == 'SLURM':
  FILECONTENT = """#!/bin/bash
#SBATCH --time=%s 
#SBATCH --output=imsrg_log/%s.%%j 
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=%d 
#SBATCH --nodelist=strongint[07-10]
#  #SBATCH --exclusive
#SBATCH --partition=all
cd $SLURM_SUBMIT_DIR
echo NTHREADS = %d
export OMP_NUM_THREADS=%d
time srun %s
"""

### Make a directory for the log files, if it doesn't already exist
if not path.exists('imsrg_log'): mkdir('imsrg_log')
if not path.exists('output'): mkdir('output')

### Loop over multiple jobs to submit
N=49 
for Z in [29]:
  for e in [14]:
     A = N + Z 
     ARGS['emax'] = '%d'%e
     ARGS['lmax'] = '%d'%e
     ARGS['e3max'] = '24'
     ARGS['A'] = '%d'%A
     ARGS['reference'] = '%s%d'%(ELEM[Z],A)
     ARGS['valence_space'] = "vs_ca48_fpgd5"
     ARGS['custom_valence_space'] = 'Ca48,p0f7,p0f5,n0f5,p1p3,n1p3,p1p1,n1p1,n0g9,n1d5'
     ARGS['denominator_delta_orbit'] = "all"
     ARGS['denominator_delta'] = 10.0 
     ARGS['BetaCM'] = "3.0" 
     ARGS['Operators'] = '' # 'GamowTeller'    # Operators to consistenly transform, separated by commas.
     #ARGS['input_op_fmt'] = 'miyagi'
     #ARGS["OperatorsFromFile"]  = "GT_2BC^1_1_0_2^/data_share13/takayuki/me2j/AxialV_Tz1-N2LO-NonLocal4-394_c3_-3.2_c4_5.4_cD_1.264_bare_hw16_emax14_e2max28.me2j.gz"

    ### Make an estimate of how much time to request. Only used for slurm at the moment.
     time_request = '999:00:00'
     if   e <  5 : time_request = '18:10:00'
     elif e <  8 : time_request = '28:00:00'
     elif e < 10 : time_request = '54:00:00'
     elif e < 12 : time_request = '92:00:00'

     jobname  = 'N%d_Z%d_%s_%s_%s_%s_e%s_E%s_s%s_hw%s_A%s' %(N, Z, ARGS['valence_space'], ARGS['LECs'],ARGS['method'],ARGS['reference'],ARGS['emax'],ARGS['e3max'],ARGS['smax'],ARGS['hw'],ARGS['A'])
     logname = jobname + datetime.fromtimestamp(time()).strftime('_%y%m%d%H%M.log')

  ### Some optional parameters that we probably want in the output name if we're using them
     if 'lmax3' in ARGS:  jobname  += '_l%d'%(ARGS['lmax3'])
     if 'eta_criterion' in ARGS: jobname += '_eta%s'%(ARGS['eta_criterion'])
     if 'core_generator' in ARGS: jobname += '_' + ARGS['core_generator']
     if 'BetaCM' in ARGS: jobname += '_' + ARGS['BetaCM']
     ARGS['flowfile'] = 'output/BCH_' + jobname + '.dat'
     ARGS['intfile']  = 'output/' + jobname

     cmd = ' '.join([exe] + ['%s=%s'%(x,ARGS[x]) for x in ARGS])

  ### Submit the job if we're running in batch mode, otherwise just run in the current shell
     if batch_mode==True:
       sfile = open(jobname+'.batch','w')
       if BATCHSYS == 'PBS':
         sfile.write(FILECONTENT%(jobname,environ['PWD'],NTHREADS,mail_address,logname,NTHREADS,cmd))
         sfile.close()
         call(['qsub', jobname+'.batch'])
       elif BATCHSYS == 'SLURM':
         sfile.write(FILECONTENT%(time_request,jobname,NTHREADS,NTHREADS,NTHREADS,cmd))
         sfile.close()
         call(['sbatch', jobname+'.batch'])
       #remove(jobname+'.batch') # delete the file
       sleep(0.1)
     else:
       call(cmd.split())  # Run in the terminal, rather than submitting


