#!/bin/bash

#SBATCH -J main_best_hier            #Job name(--job-name)
#SBATCH -o logs/slurm_%j.log          #Name of stdout output file(--output)
#SBATCH -e logs/slurm_%j.log		   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name 
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH -c 1                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=23000        # Memory per node specification is in MB. It is optional. 
#SBATCH --mail-user=bsantraigi@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
pwd; hostname; date

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cuda/10.1

# source /home/$USER/.bashrc

export CUDA_VISIBLE_DEVICES=0,1
mpirun -bootstrap slurm which python
mpirun -bootstrap slurm nvcc --version
mpirun -bootstrap slurm python main.py -embed 104 -heads 4 -hid 64 -l_e1 4 -l_e2 2 -l_d 2 -d 0.2024 -bs 8 -e 30 -model HIER

# {'nhead': 4, 'embedding_perhead': 26, 'nhid_perhead': 16, 'nlayers_e1': 4, 'nlayers_e2': 2, 'nlayers_d': 2, 'dropout': 0.20244212555189078, 'batch_size': 8, 'epochs': 5, 'model_type': 'HIER', 'embedding_size': 104, 'nhid': 64}

# [I 2020-09-08 23:24:51,165] Trial 2 finished with value: 55.70961030741921 and parameters: {'nhead': 4, 'embedding_perhead': 26, 'nhid_perhead': 16, 'nlayers_e1': 4, 'nlayers_e2': 2, 'nlayers_d': 2, 'dropout': 0.20244212555189078}. Best is trial 2 with value: 55.70961030741921.