#!/bin/bash
#SBATCH -J matlabtest                                           
#SBATCH --ntasks=8                                 
#SBATCH --nodes=1                                    
#SBATCH --ntasks-per-node=20                      
#SBATHC --time=00:30:2
#SBATCH -p low                                                
#SBATCH --output=%j.log                                     
#SBATCH --mail-user=suzs19@snnu.edu.cn
#SBATCH --mail-type=ALL
#SBATCH --output=matlab_test.out
#SBATCH --error=matlab_test.err


cd  $SLURM_SUBMIT_DIR
INPUT=lye.m
OLDDIR=`pwd`
cp $INPUT $INPUT.bak
srun hostname -s | sort -n >nodelist
EXEC_HOST=`head -n 1 nodelist`
NP=`cat nodelist | wc -l`
sed -i "s/numpar/$NP/g" $INPUT
ssh $EXEC_HOST "cd $OLDDIR ; /home/lenovo/Soft/Polyspace/R2019a/bin/matlab -nodisplay < $INPUT"
rm -rf  nodelist
