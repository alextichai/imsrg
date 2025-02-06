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
NTHREADS=24
exe = '/Users/alexandertichai/Work/Code/imsrg++/src/build/imsrg++'
exe = '/home/porro/imsrg/src/imsrg++'
#change here
### Flag to swith between submitting to the scheduler or running in the current shell
batch_mode=True
#batch_mode=True
if 'terminal' in argv[1:]: batch_mode=False

### Don't forget to change this. I don't want emails about your calculations...
mail_address = 'porro@theorie.ikp.physik.tu-darmstadt.de'

### This comes in handy if you want to loop over Z
ELEM = ['n','H','He','Li','Be','B','C','N',
       'O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K',
       'Ca','Sc','Ti','V','Cr','Mn','Fe','Co',  'Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y',
       'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',  'Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
       'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb']# ,'Bi','Po','At','Rn','Fr','Ra','Ac','Th','U','Np','Pu']

### ARGS is a (string => string) dictionary of input variables that are passed to the main program
ARGS  =  {}

### Maximum value of s, and maximum step size ds
ARGS['smax'] = '500' #'500'
ARGS['dsmax'] = '0.5'

#ARGS['lmax3'] = '10' # for comparing with Heiko

### Norm of Omega at which we split off and start a new transformation
ARGS['omega_norm_max'] = '0.25'

### Name of a directory to write Omega operators so they don't need to be stored in memory. If not given, they'll just be stored in memory.
#ARGS['scratch'] = 'SCRATCH'    

### Generator for core decoupling, can be atan, white, imaginary-time.  (atan is default)
#ARGS['core_generator'] = 'imaginary-time' 
### Generator for valence deoupling, can be shell-model, shell-model-atan, shell-model-npnh, shell-model-imaginary-time (shell-model-atan is default)
#ARGS['valence_generator'] = 'shell-model-imaginary-time' 

### Solution method
ARGS['method'] = 'magnus'
#ARGS['method'] = 'brueckner'
#ARGS['method'] = 'flow'
#ARGS['method'] = 'HF'
#ARGS['method'] = 'MP3'

### Tolerance for ODE solver if using flow solution method
#ARGS['ode_tolerance'] = '1e-5'

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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=%d
#SBATCH --partition=fast
#SBATCH --exclude=strongint14,strongint15
#SBATCH --output=imsrg_log/%s.%%j
#SBATCH --time=%s
#SBATCH --mail-user=%s
#SBATCH --mail-type=END

# Update the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64

cd $SLURM_SUBMIT_DIR
echo NTHREADS = %d
export OMP_NUM_THREADS=%d
time srun %s
"""
#SBATCH --exclude=strongint14,strongint15
#SBATCH -w strongint11,strongint13

### Make a directory for the log files, if it doesn't already exist
if not path.exists('imsrg_log'): mkdir('imsrg_log')

### Loop over multiple jobs to submit
for lda in [0.]: 
#for lda in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]:
 ARGS['polarisability'] = lda # LOOP HERE FOR POLARISABILITY
 for A in [40]:
  Z = 20 #A//2
  for reference in ['%s%d'%(ELEM[Z],A)]:
   ARGS['reference'] = reference
   print('Z = ', Z)
   if lda != 0.:
    print('lda pol = ', lda)
   for e in [4]:
    for hw in [16]:

      e3max = 16 #16  24
      #e3max = min(e3max, 3 * e)

      ARGS['emax']  = str(e)
      ARGS['e2max'] = str(2 * e)
      ARGS['e3max'] = str(e3max)

      ### Model space parameters used for reading Darmstadt-style interaction files
      #ARGS['file2e1max'] = ARGS['emax'] 
      #ARGS['file2e2max'] = ARGS['e2max']
      #ARGS['file2lmax']  = ARGS['emax'] 
      ARGS['file2e1max'] = '18 file2e2max=36 file2lmax=18'
      ARGS['file3e1max'] = '16 file3e2max=32 file3e3max=24'

      #ARGS['fmt2'] = 'me2jp'

      # Pre-contracted matrix elements
      #ARGS['1bme'] = '/home/porro/me/imsrg/%s/EM_1.8_2.0_hw%i_eMax%02d_E3Max%i.me1j.gz' %(reference, hw, e, e3max) # 1B
      #ARGS['2bme'] = '/home/porro/me/imsrg/%s/EM_1.8_2.0_hw%i_eMax%02d_E3Max%i.me2jp.gz'%(reference, hw, e, e3max) # 2B

      # Usual matrix elements
      ARGS['2bme'] = '/data_share11/takayuki/me2j/TwBME-HO_NN-only_DN2LOGO394_bare_hw%i_emax18_e2max36.me2j.gz'%(hw)    # 2B
      ARGS['3bme'] = '/data_share11/takayuki/me3j/NO2B_ThBME_DNNLOgo_3NFJmax15_IS_hw%i_ms16_32_24.stream.bin'%(hw)      # 3B, hw = 16 MeV

      ARGS['3bme_type'] = 'no2b'

      #ARGS['LECs'] = ''
      #ARGS['3bme_type'] = 'no2b' # 'full'

      ARGS['moments'] = 'true'
      ARGS['isospin_ch'] = 'isoscalar'

      ARGS['write_Hamiltonian'] = 'false'

      ARGS['hw']   = '%d'%hw
      ARGS['A']    = '%d'%A

      ARGS['valence_space'] = reference
      # ARGS['valence_space'] = '0hw-shell'
      # ARGS['valence_space'] = 'Cr%d'%A
      # ARGS['core_generator'] = 'imaginary-time'
      # ARGS['valence_generator'] = 'shell-model-imaginary-time'
      
      # ARGS['method'] = method

      # ARGS['Operators'] = ''    # Operators to consistenly transform, separated by commas.
      # ARGS['Operators'] = 'Rp2'
      # ARGS['Operators'] = 'Rm2lab' # which other operators to coevolve and transform
      # ARGS['Operators'] = 'E2,M1'

      ### Make an estimate of how much time to request. Only used for slurm at the moment.
      time_request = '10-00:00:00'
      #if   e <  5 : time_request = '00:10:00'
      #elif e <  8 : time_request = '01:00:00'
      #elif e < 10 : time_request = '04:00:00'
      #elif e < 12 : time_request = '12:00:00'
      #elif e < 14 : time_request = '24:00:00'

      jobname  = '%s_%s_%s_e%s_E%s_s%s_hw%s_A%s' %(ARGS['valence_space'],ARGS['method'],ARGS['reference'],ARGS['emax'],ARGS['e3max'],ARGS['smax'],ARGS['hw'],ARGS['A'])
      logname = jobname + datetime.fromtimestamp(time()).strftime('_%y%m%d%H%M.log')

      ### Some optional parameters that we probably want in the output name if we're using them
      if 'lmax3' in ARGS:  jobname  += '_l%d'%(ARGS['lmax3'])
      if 'eta_criterion' in ARGS: jobname += '_eta%s'%(ARGS['eta_criterion'])
      if 'core_generator' in ARGS: jobname += '_' + ARGS['core_generator']
      if 'BetaCM' in ARGS: jobname += '_' + ARGS['BetaCM']
      if ARGS['polarisability'] != 0.: jobname += '_lda%f'%ARGS['polarisability']
      ARGS['flowfile'] = 'output/BCH_' + jobname + '.dat'
      ARGS['intfile']  = '/data_share11/ME_IMSRG/' + jobname #

      cmd = ' '.join([exe] + ['%s=%s'%(x,ARGS[x]) for x in ARGS])

      ### Submit the job if we're running in batch mode, otherwise just run in the current shell
      if batch_mode==True:
        sfile = open(jobname+'.batch','w')
        if BATCHSYS == 'PBS':
          sfile.write(FILECONTENT%(jobname,environ['PWD'],NTHREADS,mail_address,logname,NTHREADS,cmd))
          sfile.close()
          call(['qsub', jobname+'.batch'])
        elif BATCHSYS == 'SLURM':
          sfile.write(FILECONTENT%(NTHREADS,jobname,time_request,mail_address,NTHREADS,NTHREADS,cmd))
          sfile.close()
          call(['sbatch', jobname+'.batch'])
        remove(jobname+'.batch') # delete the file
        sleep(0.1)
      else:
        call(cmd.split())  # Run in the terminal, rather than submitting

