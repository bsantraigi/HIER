#!/bin/bash

#SBATCH -J HIER_marco            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log		   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name 
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 1                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=23000        # Memory per node specification is in MB. It is optional. 
##SBATCH --mail-user=bsantraigi@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH -w gpu016
pwd; hostname; date

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cuda/10.1

source /home/$USER/.bashrc

export CUDA_VISIBLE_DEVICES=0
nvidia-smi
mpirun -bootstrap slurm which python
mpirun -bootstrap slurm nvcc --version
mpirun -bootstrap slurm python -u train_generator.py --option train --model model/ --batch_size 384 --max_seq_length 50 --act_source bert --learning_rate 1e-4 --nlayers_e 6 --nlayers_d 3 --seed 0
# mpirun -bootstrap slurm python -u train_generator.py --option train --model model/ --batch_size 512 --max_seq_length 50 --act_source bert --learning_rate 1e-4 --nlayers_e 3 --nlayers_d 3 --seed 0
