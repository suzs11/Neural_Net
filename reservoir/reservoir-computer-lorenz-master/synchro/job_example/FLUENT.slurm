#!/bin/bash 
#SBATCH -J fluent_test
#SBATCH -p low
#SBATCH -n 240
#SBATCH -N 10
#SBATCH --ntasks-per-node=24
#SBATCH -o out.%j
#SBATCH -e err.%j

#########################################################
### slurm script for fluent18.2 on Sugon's silicon cube supercomputer
### version 1.0 | 2017-10-11 | coded by Zhang Guoliang
#########################################################
### Please input your job parameters as follows:
#########################################################
EXEC=/public/software/apps/ansys182/v182/fluent/bin/fluent  
WORK_DIR=/public/home/zhanggl/fluent_case/
#WORK_DIR=`pwd`
FLUENT_VERSION=3d  ## [ 3d | 3ddp | 2d | 2ddp ] 
INPUT_FILE=${WORK_DIR}/case.jou      
LOG_FILE=${WORK_DIR}/case.log   
### optional and advance parameters
#OPTIONAL=" "
OPTIONAL="-platform=intel -mpi=intel -ssh"   
#########################################################

cd $WORK_DIR  

PROCS=$SLURM_NPROCS

#HOST_FILE=$(generate_pbs_nodefile)

export PBS_NODEFILE=`generate_pbs_nodefile`  
#HOST_FILE=$PBS_NODEFILE 
sort $PBS_NODEFILE | uniq -c | awk '{print $2":"$1}' > ./hostfile_$PROCS 
HOST_FILE=./hostfile_$PROCS

echo The start time is: `date +"%Y-%m-%d %H:%M:%S"` | tee $LOG_FILE  
echo My job ID is: $SLURM_JOB_ID | tee -a $LOG_FILE 
echo The total cores is: $PROCS | tee -a $LOG_FILE 
echo The hosts is: | tee -a $LOG_FILE 
cat $HOST_FILE | tee -a $LOG_FILE

$EXEC -g $FLUENT_VERSION -t$PROCS -cnf=$HOST_FILE -i $INPUT_FILE $OPTIONAL 2>&1 | tee -a  $LOG_FILE

echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE  