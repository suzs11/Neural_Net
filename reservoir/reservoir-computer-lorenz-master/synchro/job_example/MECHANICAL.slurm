#!/bin/bash 
#SBATCH -J RP_JOB_NAME
#SBATCH -p RP_QUEUE
#SBATCH -n RP_NPROC
#SBATCH -N RP_NNODE
#SBATCH --ntasks-per-node=RP_PPN
#SBATCH --gres=gpu:RP_NGPU
#SBATCH -w RP_NODE_STRING
#SBATCH --time RP_WALL_TIME 
#SBATCH --comment=MECHANICAL
#SBATCH -o std.out.%j
#SBATCH -e std.err.%j

### MARK_LICENCE   ###  #PBS -l software=login1@1055%cfd_2:1

#########################################################
### Slurm scripts for Sugon Portal_5.0 of Ansys_Mechanical_19.x/18.x/17.x   ### slurm2pbs  
### Version 1.0    |  2019-09-25  |  created by Zhang Guoliang
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

INPUT_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.inp
OUTPUT_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.out
LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma

HOST_STRING=`srun hostname | sort | uniq -c | awk '{print $2":"$1}' | tr '\n' ':' | sed 's/:$//'` ### slurm2pbs
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

### Create input/inp file

if [ "$INPUT_TYPE" == "db" ];then
  DB_NAME=$(basename $DB_FILE)
  DB_NAME=${DB_NAME%.*}
cat <<EOF > $INPUT_FILE
resume,${DB_NAME},db   
/solu 
SOLVE 
y  
save,all 
finish  
EOF

elif [ "$INPUT_TYPE" == "inp" ];then
  dos2unix $INP_FILE >& /dev/null
  INPUT_FILE=$INP_FILE

else
  echo "Error: wrong value of \$INPUT_TYPE : $INPUT_TYPE" && exit 1 
fi

### Run APP

## !!! for Mechanical,strongly suggest closing GUI in most cases 

if [[ -z $NGPU ]] || [[ $NGPU -eq 0 ]];then OPT_GPU="" ; fi
if [[ -n $NGPU ]] && [[ $NGPU -ge 1 ]];then OPT_GPU="-acc nvidia -na $NGPU" ; fi

cd $WORK_DIR
$APP $OPT_GUI -b -dis -machines $HOST_STRING -dir $WORK_DIR -i $INPUT_FILE -o $OUTPUT_FILE -p$NET_TYPE $OPT_GPU $CMD_OPTION_ADMIN $CMD_OPTION_USER $CMD_OPTION_WEB 2>&1 | tee -a $LOG_FILE 

echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE  
 
