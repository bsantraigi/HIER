#!/bin/bash

#SBATCH -J hier_search            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log		   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name 
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 1                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=23000        # Memory per node specification is in MB. It is optional. 
#SBATCH --mail-user=bishal.santra@iitkgp.ac.in        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --time 2-0

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cuda/10.1

# source /home/$USER/.bashrc

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
mpirun -bootstrap slurm which python
mpirun -bootstrap slurm nvcc --version
mpirun -bootstrap slurm python search_params_acts.py -e 5 -model HIER++
#mpirun -bootstrap slurm python search_params_acts.py -e 5 -model SET++
