#!/bin/bash

#SBATCH -J main_best_hier            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log		   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name 
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 1                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=6000        # Memory per node specification is in MB. It is optional. 
#SBATCH --mail-user=bsantraigi@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
pwd; hostname; date

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cuda/10.1

# source /home/$USER/.bashrc

export CUDA_VISIBLE_DEVICES=0,1
mpirun -bootstrap slurm which python
mpirun -bootstrap slurm nvcc --version

mpirun -bootstrap slurm python main_acts.py -embed 175 -heads 7 -hid 91 -l_e1 4 -l_e2 6 -l_d 3 -d 0.071 -bs 16 -e 60 -model HIER++
# mpirun -bootstrap slurm python main_acts.py -embed 196 -heads 7 -hid 98 -l_e1 2 -l_e2 4 -l_d 6 -d 0.001 -bs 8 -e 30 -model HIER++
# mpirun -bootstrap slurm python main_acts.py -embed 175 -heads 7 -hid 91 -l_e1 4 -l_e2 6 -l_d 3 -d 0.071 -bs 8 -e 30 -model SET++

# HIER++
# {'nhead': 7, 'embedding_perhead': 28, 'nhid_perhead': 14, 'nlayers_e1': 2, 'nlayers_e2': 4, 'nlayers_d': 6, 'dropout': 0.001088208050525544, 'batch_size': 8, 'epochs': 5, 'model_type': 'HIER++', 'embedding_size': 196, 'nhid': 98, 'log_path': 'running/transformer_hier++/'}

# [I 2020-09-23 08:04:10,493] Trial 22 finished with value: 108.47518434207636 and parameters: {'nhead': 7, 'embedding_perhead': 28, 'nhid_perhead': 14, 'nlayers_e1': 2, 'nlayers_e2': 4, 'nlayers_d': 6, 'dropout': 0.001088208050525544}. Best is trial 22 with value: 108.47518434207636.

# SET++
# {'nhead': 7, 'embedding_perhead': 25, 'nhid_perhead': 13, 'nlayers_e1': 4, 'nlayers_e2': 6, 'nlayers_d': 3, 'dropout': 0.07106661193491001, 'batch_size': 8, 'epochs': 5, 'model_type': 'SET++', 'embedding_size': 175, 'nhid': 91}

# [I 2020-09-15 23:23:07,025] Trial 10 finished with value: 90.55630105889372 and parameters: {'nhead': 7, 'embedding_perhead': 25, 'nhid_perhead': 13, 'nlayers_e1': 4, 'nlayers_e2': 6, 'nlayers_d': 3, 'dropout': 0.07106661193491001}. Best is trial 10 with value: 90.55630105889372.