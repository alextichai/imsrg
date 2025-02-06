#!/usr/bin/env python

##########################################################################
##  write_H.py
######################################################################

from os import path,environ,mkdir,remove
from sys import argv
from subprocess import call,PIPE
from time import time,sleep
from datetime import datetime
import glob
import re

### Check to see what type of batch submission system we're dealing with
BATCHSYS = 'NONE'
if call('type '+'qsub', shell=True, stdout=PIPE, stderr=PIPE) == 0: BATCHSYS = 'PBS'
elif call('type '+'srun', shell=True, stdout=PIPE, stderr=PIPE) == 0: BATCHSYS = 'SLURM'

### The code uses OpenMP and benefits from up to at least 24 threads
NTHREADS=24

exe = '/home/porro/imsrg/src/imsrg++'

### Flag to swith between submitting to the scheduler or running in the current shell
batch_mode=True
if 'terminal' in argv[1:]: batch_mode=False

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
ARGS['smax']  = '0' # For writing purposes (PGCM) we don't want to evolve the Hamiltonian
ARGS['dsmax'] = '0.5'

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

Nmax_arr = 1001 # depends on cluster configuration, 1001 jobs max in array for strongint

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
for A in [40]:
  Z = 20 #A//2
  for reference in ['%s%d'%(ELEM[Z],A)]:
    ARGS['reference'] = reference
    print('Reference = ', reference)
    for e in [4,6,8,10]:
      for hw in [16]:
        ARGS['emax']  = '%d' % e

        e3max = 24 #16  24
        e3max = min(e3max, 3 * e)

        smax_Omega = '500'

        ARGS['emax']  = str(e)
        ARGS['e2max'] = str(2 * e)
        ARGS['e3max'] = str(e3max)

        ### Model space parameters used for reading Darmstadt-style interaction files
        #ARGS['file2e1max'] = '18 file2e2max=36 file2lmax=18'
        #ARGS['file2e1max'] = '4 file2e2max=8 file2lmax=4'
        #ARGS['file3e1max'] = '16 file3e2max=32 file3e3max=24'
        
        intlabel = 'EM_7.5'
        # intlabel = 'EM_1.8_2.0'
        # intlabel = 'DN2LO_GO_394'

        ARGS['fmt2'] = 'me2jp'

        ARGS['file2e1max'] = ARGS['emax'] 
        ARGS['file2e2max'] = ARGS['e2max']
        ARGS['file2lmax']  = ARGS['emax']

        # Pre-contracted matrix elements
        ARGS['1bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d_E3Max%02d.me1j.gz' % (reference, intlabel, hw, e, e3max) # 1B
        ARGS['2bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d_E3Max%02d.me2jp.gz'% (reference, intlabel, hw, e, e3max) # 2B

        ARGS['omefile'] = '/home/porro/Omega/%s/%s_hw%i_eMax%02d_E3Max%02d_s%s' % (reference, intlabel, hw, e, e3max, smax_Omega)

        ARGS['hw']   = '%d'%hw
        ARGS['A']    = '%d'%A

        ARGS['valence_space'] = reference

        # Select the desired response here ############
        L = 1

        ARGS['L_MixMom']   = L
        ARGS['isospin_ch'] = 'isovector'

        ###############################################

        files = glob.glob(ARGS['omefile'] + "*.me2jp.gz")

        nmagn = []
        for f in files:
            match = re.search(r's500_(\d+)', f)
            if match:
                nmagn.append(int(match.group(1)))  # Only add if a match is found

        if nmagn:
            N = len(nmagn)

        input(f"{N} omegas found")

        ARGS['N_Magnus'] = N

        ARGS['kernel'] = 'true'

        # smax of the calculation we are starting from, just for naming purposes
        smax_prev = '500'

        ### Make an estimate of how much time to request. Only used for slurm at the moment.
        time_request = '10-00:00:00'
        #if   e <  5 : time_request = '00:10:00'
        #elif e <  8 : time_request = '01:00:00'
        #elif e < 10 : time_request = '04:00:00'
        #elif e < 12 : time_request = '12:00:00'
        #elif e < 14 : time_request = '24:00:00'

        # Loop over qs

        qmin = 0.3
        qmax = 3.3

        dq = 0.3

        N = int((qmax - qmin) / dq)

        qs = []

        qs.append(0.)     # Add full q = 0 limit 
        # qs.append(0.001)  # and something close to check (no prefactor problem to match the long wavelength limit)

        for i in range(N):
          qi = qmin + i * dq
          qs.append(qi)

        Nker = int(len(qs) * (len(qs) + 1) / 2)

        input (f"{Nker} kernels are going to be evaluated, do you want to continue ?")

        # add check on existing kernel

        for qL in qs:
          for qR in qs:
            if (qR > qL):
            #if (qR != qL):
              continue
             
            ARGS['qL'] = qL
            ARGS['qR'] = qR

            #jobname  = '%s_%s_%s_%s_e%s_E%s_s%s_hw%s_A%s' %(ARGS['valence_space'], ARGS['LECs'],ARGS['method'],ARGS['reference'],ARGS['emax'],ARGS['e3max'],ARGS['smax'],ARGS['hw'],ARGS['A'])
            jobname  = '%s_kernel_%s_e%s_E%s_s%s_hw%s' %(ARGS['valence_space'],intlabel,ARGS['emax'],ARGS['e3max'],ARGS['smax'],ARGS['hw'])
            logname = jobname + datetime.fromtimestamp(time()).strftime('_%y%m%d%H%M.log')

            ### Make a directory for the output (kernels), if it doesn't already exist
            kernel_dir = '/home/porro/kernels_imsrg/%s_%s_hw%s_eMax%02d_E3Max%s_s%s' % (ARGS['valence_space'], intlabel, ARGS['hw'], int(ARGS['emax']), ARGS['e3max'], smax_prev)
            if not path.exists(kernel_dir): mkdir(kernel_dir)

            ARGS['kerdir'] = kernel_dir

            kername = '%s/L=%i_%s_%.3f_%.3f.dat' % (kernel_dir, L, ARGS['isospin_ch'], qL, qR)
            #if path.exists(kername):
            #  continue

            #ARGS['intfile']  = '/data_share11/ME_IMSRG/' + jobname
            #ARGS['intfile'] = out_dir   + '/%s_hw%s_eMax%02d_E3Max%s'     % (intlabel, ARGS['hw'], int(ARGS['emax']), ARGS['e3max'])
            
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
