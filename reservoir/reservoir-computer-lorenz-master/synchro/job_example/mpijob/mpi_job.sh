#!/bin/sh
#SBATCH --job-name=OmpJob
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=debug

# load the environment
module purge      #clean the environment
module load mpi/intelmpi/2017.4.239
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so
# running the command
#mpiicc -g -pthread -o main.x main.c
srun main.x

