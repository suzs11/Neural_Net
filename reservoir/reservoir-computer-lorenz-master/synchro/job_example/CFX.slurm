#!/bin/bash 
#SBATCH -J RP_JOB_NAME
#SBATCH -p RP_QUEUE
#SBATCH -n RP_NPROC
#SBATCH -N RP_NNODE
#SBATCH --ntasks-per-node=RP_PPN
#SBATCH --gres=gpu:RP_NGPU
#SBATCH -w RP_NODE_STRING
#SBATCH --time RP_WALL_TIME 
#SBATCH --comment=CFX
#SBATCH -o std.out.%j
#SBATCH -e std.err.%j

### MARK_LICENCE   ###  #PBS -l software=login1@1055%cfd_2:1

#########################################################
### Slurm scripts for Sugon Portal_5.0 of CFX_19.x/18.x/17.x   ### slurm2pbs  
### Version 1.0    |  2019-11-08  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-04-02  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

source ~/.job_portal.var
source ~/.job_interface.var

### Set basic var   ### MARK_slurm2pbs

JOBID=$SLURM_JOB_ID                                  ### slurm2pbs
QUEUE=$SLURM_JOB_PARTITION                           ### slurm2pbs
NP=$SLURM_NPROCS                                     ### slurm2pbs
NNODE=`srun hostname | sort | uniq | wc -l`          ### slurm2pbs

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma

HOST_STRING=`srun hostname | sort | uniq -c | awk '{print $2"*"$1}' | tr '\n' ',' | sed 's/,$//'`  ### slurm2pbs
echo $HOST_STRING > $HOST_FILE                       ### slurm2pbs

### Create GUI   ### MARK_GUI_START

if [ $GUI -eq 1 ];then 
  source /usr/local/vnc/conf/vncfunction
  if [ $GV_TYPE == HPC ];then 
    session_id=`vnc_start $JOBID $QUEUE` 
  elif [ $GV_TYPE == IndustrialCloud ];then
    session_id=`vnc_start -u $USER -j $JOBID -q $QUEUE -s /usr/bin/xterm|grep ^SID`
  else
    echo "ERROR: wrong value of \$GV_TYPE : $GV_TYPE" && exit 1  
  fi  
  PROXY_HOSTNAME=$(echo $session_id |cut -d',' -f 10 | awk -F= '{print $2}')
  PROXY_SID=$(echo $session_id |cut -d',' -f 1 | awk -F= '{print $2}')
  RET=$?
  if [ $RET -ne 0 ]; then
    echo "start bad, job failed to start" | tee -a $LOG_FILE 
    exit 1
  fi
  export DISPLAY=${PROXY_HOSTNAME}:${PROXY_SID}
  trap "source /usr/local/vnc/conf/vncfunction; vnc_stop $PROXY_SID $PROXY_HOSTNAME" 1 2 9 15
  trap "source /usr/local/vnc/conf/vncfunction; vnc_stop $PROXY_SID $PROXY_HOSTNAME" EXIT 
fi
### MARK_GUI_END

### Write basic job infomations

echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n"  | tee -a $LOG_FILE 

### Run APP

#if [ $GUI -eq 0 ];then OPT_GUI="" ; fi
#if [ $GUI -eq 1 ];then OPT_GUI="xx" ; fi

#if [ $NGPU -ge 1 ];then OPT_GPU="xx" ; fi

if [ "$PRECISSION" == 'single' ];then OPT_PRECISSION="" ; fi
if [ "$PRECISSION" == 'double' ];then OPT_PRECISSION="-double" ; fi

if [ -z $DEF_FILE ];then OPT_DEF="" ; fi
if [ -n $DEF_FILE ];then OPT_DEF="-continue-from-file $DEF_FILE -use-mesh-from-iv" ; fi

cd $WORK_DIR
export CFX5RSH=ssh    ## value = ssh | rsh
if [ $GUI -eq 0 ];then
  $APP $OPT_GUI $OPT_GPU $OPT_PRECISSION -par-dist $HOST_STRING -start-method 'Intel MPI Distributed Parallel' -def $DEF_FILE $OPT_DEF 2>&1 | tee -a $LOG_FILE
elif [ $GUI -eq 1 ];then
  $APP 
else
  echo "ERROR: wrong value of \$APP : $APP" | tee -a $LOG_FILE
fi

echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE
