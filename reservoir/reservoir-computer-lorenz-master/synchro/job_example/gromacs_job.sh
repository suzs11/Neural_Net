#!/bin/bash
#SBATCH --job-name=gromacs
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=debug

# load the environment
module purge
module load compiler/intel/composer_xe_2017.5.239
module load mpi/intelmpi/2017.4.239
source /public/software/compiler/intel/intel-compiler-2017.5.239/mkl/bin/mklvars.sh intel64
module load compiler/gnu/6.4.0
source /public/software/apps/gromacs/2019.5/intelmpi/bin/GMXRC

gmx_mpi grompp -f em.mdp -c decane.pdb -p decane.top -o em.tpr
mpirun -np 8 gmx_mpi mdrun -s em.tpr -deffnm decane_min -c decane_min.pdb -nb cpu
