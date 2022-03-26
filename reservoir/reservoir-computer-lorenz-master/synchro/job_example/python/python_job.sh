#!/bin/bash
#SBATCH --job-name=PythonTest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=debug
#SBATCH --time=20:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=suzs@163.com


# load the environment
module purge
module load apps/python/3.6.1

# run python
python --version

