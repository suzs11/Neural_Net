#!/bin/bash
#SBATCH -J fsi-p16_0
#SBATCH -p silicon_m
#SBATCH -N 2
#SBATCH -n 32
#SBATCH --ntasks-per-node=16
#SBATCH -o %j.out
#SBATCH -e %j.err

###############################################
### user interface 
###############################################
APP_EXEC=/public/software/ansys_inc/v160/autodyn/bin/autodyn160
WORK_DIR=/public/home/duxw/cae-wdir/autodyn/u_ansiyatai_2016-3-21/fsi-p32_0/
INPUT_FILE=fsi-p32_0.ad
OUTPUT_LOG=output.log

NP=$SLURM_NPROCS
HOST_FILE=$(generate_pbs_nodefile)

var_PPDEF="parallel"
var_sp=1000
var_mem=1024

###############################################
### generate parallel.cfg
###############################################
var_EPDEF=${APP_EXEC%/*}/linx64/
#var_PPDEF=defined_above
var_PPCFG=${var_PPDEF}
#var_sp=defined_above
#var_mem=defined_above
#var_cpu=defined_below
#var_task=defined_below

cd ${WORK_DIR}
mv parallel.cfg parallel.cfg.bak
echo "#@EPDEF=${var_EPDEF}" > parallel.cfg 
echo "#@PPDEF ${var_PPDEF}" >> parallel.cfg 
echo "#@PPCFG ${var_PPCFG}" >> parallel.cfg
cat ${HOST_FILE} > ./HOST_FILE
sort -u ${HOST_FILE} > ./HOST_FILE_uniq
sum_node=`cat ./HOST_FILE_uniq | wc -l` 
for i in `seq 1 ${sum_node}`; do \
  node_i=`sed -n "${i}p" ./HOST_FILE_uniq`
  node_i_np=`cat ${HOST_FILE} | grep ${node_i} | wc -l`
  var_cpu=${node_i_np}
  var_task=${node_i_np}
  echo "${node_i} sp=${var_sp}" >> parallel.cfg
  echo "#@ mem=${var_mem} cpu=${var_cpu} task=${var_task}" >> parallel.cfg
done

###############################################
### run app
###############################################
cd ${WORK_DIR}
${APP_EXEC} -I ${INPUT_FILE} | tee $OUTPUT_LOG


