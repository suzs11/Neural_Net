#!/bin/bash
#SBATCH --job-name=VaspTest
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal

# load the environment
module purge
source /public/software/profile.d/compiler_intel-compiler-2017.5.239.sh
source /public/software/profile.d/mpi_intelmpi-2017.4.239.sh
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so
export PATH=/public/software/apps/vasp/5.4.4/intelmpi:${PATH}

srun vasp_std
