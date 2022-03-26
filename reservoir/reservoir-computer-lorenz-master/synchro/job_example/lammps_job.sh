#!/bin/bash
#SBATCH --job-name=LAMMPS
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.log
#SBATCH --partition=normal

# load the environment
module purge
source /public/software/compiler/intel/intel-compiler-2017.5.239/bin/compilervars.sh intel64
source /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/bin/mklvars.sh intel64
source /public/software/mpi/intelmpi/2017.4.239/bin64/mpivars.sh
export PATH=/public/software/apps/lammps/7Aug19/intelmpi:${PATH}
export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so

srun lmp_intelmpi -v x 32 -v y 32 -v z 32 -v t 100 < in.lj
