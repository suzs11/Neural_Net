#!/bin/sh
#SBATCH --job-name=SerialJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug

# load the environment
module purge      #clean the environment
# running the command
echo hello world

