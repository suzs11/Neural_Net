#!/bin/bash
#SBATCH -J matlabtest                                           
#SBATCH --ntasks=8                                 
#SBATCH --nodes=1                                    
#SBATCH --ntasks-per-node=8                      
#SBATCH -p low                                                
#BATCH --output=%j.log                                     

cd  $SLURM_SUBMIT_DIR
INPUT=lye.m
OLDDIR=`pwd`
cp $INPUT $INPUT.bak
srun hostname -s | sort -n >nodelist
EXEC_HOST=`head -n 1 nodelist`
NP=`cat nodelist | wc -l`
sed -i "s/numpar/$NP/g" $INPUT
ssh $EXEC_HOST "cd $OLDDIR ; /public/software/matlab2019/bin/matlab -nodisplay < $INPUT"
rm -rf  nodelist
