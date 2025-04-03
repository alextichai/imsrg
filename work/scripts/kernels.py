#!/usr/bin/env python

##########################################################################
##  write_H.py
######################################################################

from os import path,environ,mkdir,remove
from sys import argv
from subprocess import call,PIPE
from time import time,sleep
from datetime import datetime
from collections import namedtuple
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
# for beta in [0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75]:
for A in [4]:
  Z = 2
  for reference in ['%s%d'%(ELEM[Z],A)]:
    ARGS['reference'] = reference
    print('Reference = ', reference)
    for e in [4]:
      for hw in [25]:
        ARGS['emax']  = '%d' % e

        e3max = 24 #16  24
        e3max = min(e3max, 3 * e)

        smax_Omega = '500'

        ARGS['emax']  = str(e)
        ARGS['e2max'] = str(2 * e)

        twobody = True # Set to True for calculations WITHOUT 3b forces

        if twobody == False:
          ARGS['e3max'] = str(e3max)
        
        # intlabel = 'EM_7.5'
        # intlabel = 'EM_1.8_2.0'
        # intlabel = 'DN2LO_GO_394'
        # intlabel = 'NNLO_sat'
        intlabel = 'N2LO_opt'

        ARGS['fmt2'] = 'me2jp'

        # ARGS['BetaCM'] = '%f' %beta

        ARGS['file2e1max'] = ARGS['emax'] 
        ARGS['file2e2max'] = ARGS['e2max']
        ARGS['file2lmax']  = ARGS['emax']

        # Pre-contracted matrix elements
        if twobody == True:
          ARGS['1bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d' % (reference, intlabel, hw, e) # 1B
          ARGS['2bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d' % (reference, intlabel, hw, e) # 2B
        else:
          ARGS['1bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d_E3Max%02d' % (reference, intlabel, hw, e, e3max) # 1B
          ARGS['2bme'] = '/home/porro/me/imsrg/%s/%s_hw%i_eMax%02d_E3Max%02d' % (reference, intlabel, hw, e, e3max) # 2B

        if twobody == True:
          ARGS['omefile'] = '/home/porro/Omega/%s/%s_hw%i_eMax%02d_s%s' % (reference, intlabel, hw, e, smax_Omega)
        else:
          ARGS['omefile'] = '/home/porro/Omega/%s/%s_hw%i_eMax%02d_E3Max%02d_s%s' % (reference, intlabel, hw, e, e3max, smax_Omega)

        if 'BetaCM' in ARGS: 
          ARGS['1bme']    += '_' + ARGS['BetaCM']
          ARGS['2bme']    += '_' + ARGS['BetaCM']
          ARGS['omefile'] += '_' + ARGS['BetaCM']

        ARGS['1bme'] += '.me1j.gz'
        ARGS['2bme'] += '.me2jp.gz'

        ARGS['hw']   = '%d'%hw
        ARGS['A']    = '%d'%A

        ARGS['valence_space'] = reference

        # Select the desired response here ############
        L = 1

        probe = 'EM' # Alternatives: 'isoscalar', 'isovector', 'EM'

        ARGS['L_MixMom']   = L
        # ARGS['isospin_ch'] = 'isoscalar'

        ###############################################

        files = glob.glob(ARGS['omefile'] + "*.me2jp.gz")

        nmagn = []
        for f in files:
          if twobody == True:
            match = re.search(r's500_(\d)', f)
          else:
            match = re.search(r's500_(\d)_(\d)', f)
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

        qmin = 0.00
        qmax = 3.00

        dq = 0.25

        N = int((qmax - qmin) / dq) + 1

        ops = []

        Operator = namedtuple("Operator", ["lda", "q"])

        if probe == 'EM':
          ops.append(Operator("0E", -2)) # Isovector operator at q = 0
          ops.append(Operator("0E", -1)) # Isoscalar operator at q = 0
          ops.append(Operator("0E",  0)) # Excitation operator at q = 0
          
          if L == 0:
            for i in range(N):
              qi = qmin + i * dq
              ops.append(Operator("C", qi))   # Only Coulomb component for the monopole
          else:
            for i in range(N):
              qi = qmin + i * dq
              ops.append(Operator("C", qi))   # Coulomb component
              ops.append(Operator("TE", qi))  # Transverse Electric component
        
        elif probe == 'isoscalar':
          # ops.append(Operator("IS", 0.01))
          # ops.append(Operator("IS", 0.05))
          # ops.append(Operator("IS", 0.10))
          # ops.append(Operator("IS", 0.15))
          
          # for i in range(N):
          #   qi = qmin + i * dq
          #   ops.append(Operator("IS", qi))   # Only Coulomb component for the monopole
          for i in range(N):
            qi = qmin + i * dq
            ops.append(Operator("PS", qi))

        elif probe == 'isovector':
          for i in range(N):
            qi = qmin + i * dq
            ops.append(Operator("IV", qi))

        # qs = []

        # qs.append(0.)     # Add full q = 0 limit

        # qs.append(98)     # Add offset m = (100 - N) to the electric multipole operator
        # qs.append(96)     # 
        # qs.append(94)     # 
        # qs.append(92)     # 
        # qs.append(90)     # 

        ### qs.append(0.001)  # and something close to check (no prefactor problem to match the long wavelength limit)

        # for i in range(N):
        #   qi = qmin + i * dq
        #   qs.append(qi)

        Nker = int(len(ops) * (len(ops) + 1) / 2)

        input (f"{Nker} kernels are going to be evaluated, do you want to continue ?")

        # add check on existing kernel

        for ll, opL in enumerate(ops):
          for rr, opR in enumerate(ops):
            qL = opL.q
            qR = opR.q
            fL = opL.lda
            fR = opR.lda

            if (rr > ll):
            #if (qR != qL):
              continue
            
            ARGS['qL'] = qL
            ARGS['qR'] = qR
            ARGS['fL'] = fL
            ARGS['fR'] = fR

            if twobody == True:
              jobname  = '%s_kernel_%s_e%s_s%s_hw%s' %(ARGS['valence_space'],intlabel,ARGS['emax'],ARGS['smax'],ARGS['hw'])
            else:
              jobname  = '%s_kernel_%s_e%s_E%s_s%s_hw%s' %(ARGS['valence_space'],intlabel,ARGS['emax'],ARGS['e3max'],ARGS['smax'],ARGS['hw'])
            logname = jobname + datetime.fromtimestamp(time()).strftime('_%y%m%d%H%M.log')

            ### Make a directory for the output (kernels), if it doesn't already exist
            if twobody == True:
              kernel_dir = '/home/porro/kernels_imsrg/%s_%s_hw%s_eMax%02d_s%s' % (ARGS['valence_space'], intlabel, ARGS['hw'], int(ARGS['emax']), smax_prev)
            else:
              kernel_dir = '/home/porro/kernels_imsrg/%s_%s_hw%s_eMax%02d_E3Max%s_s%s' % (ARGS['valence_space'], intlabel, ARGS['hw'], int(ARGS['emax']), ARGS['e3max'], smax_prev)
            if not path.exists(kernel_dir): mkdir(kernel_dir)

            ARGS['kerdir'] = kernel_dir

            # kername = '%s/L=%i_%s_%s_%.3f_%s_%.3f.dat' % (kernel_dir, L, ARGS['isospin_ch'], fL, qL, fR, qR)
            kername = '%s/L=%i_%s_%.3f_%s_%.3f.dat' % (kernel_dir, L, fL, qL, fR, qR)
            # if path.exists(kername):
            #   continue

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
