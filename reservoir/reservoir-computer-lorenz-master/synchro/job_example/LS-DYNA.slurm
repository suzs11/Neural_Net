#!/bin/bash 
#SBATCH -J lsdyna
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 2
#SBATCH --ntasks-per-node=24
#SBATCH -o out.%j
#SBATCH -e err.%j

#########################################################
### slurm script for Ansys LS-DYNA 16.0 on Sugon's 
### silicon cube supercomputer
### version 1.0 | 2018-08-15 | coded by Zhang Guoliang
#########################################################
### Please input your job parameters as follows:
#########################################################
EXEC=/public/software/apps/ansys160/v160/ansys/bin/ansys160 
WORK_DIR=/public/home/duxw/zhanggl/test-clusmon/3cars/
#WORK_DIR=`pwd`
INPUT_FILE=${WORK_DIR}/3cars_shell2_150ms.k      
OUTPUT_FILE=${WORK_DIR}/case.out  
LOG_FILE=${WORK_DIR}/case.log   
### optional and advance parameters
#OPTIONAL=" "
OPTIONAL=" -platform=intel -mpi=intel -ssh "   
#########################################################

cd $WORK_DIR  

PROCS=$SLURM_NPROCS
export PBS_NODEFILE=`generate_pbs_nodefile` 
HOST_STRING=`sort $PBS_NODEFILE | uniq -c | awk '{print $2":"$1}' | tr '\n' ':' | sed 's/:$//'`

echo The start time is: `date +"%Y-%m-%d %H:%M:%S"` | tee $LOG_FILE  
echo My job ID is: $SLURM_JOB_ID | tee -a $LOG_FILE 
echo The total CPU cores is: $PROCS | tee -a $LOG_FILE 
echo The hosts is: | tee -a $LOG_FILE 
echo $HOST_STRING | tee -a $LOG_FILE

$EXEC memory=500m -lsdynampp -dis -machines $HOST_STRING i=$INPUT_FILE o=$OUTPUT_FILE 2>&1 | tee -a $LOG_FILE

echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE  


