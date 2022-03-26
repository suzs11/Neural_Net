#!/bin/sh
#SBATCH --job-name=OmpJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=debug

# load the environment
module purge      #clean the environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# running the command
#g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main.x main.cpp
./main.x

