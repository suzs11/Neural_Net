#!/bin/bash
#SBATCH -J perf-8
#SBATCH -p silicon
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
unset SLURM_GTIDS

APP_EXEC=/public/software/apps/ansys17/ansys_inc/v170/CFX/bin/cfx5solve
WORK_DIR=/public/home/sugon_test/Test_US/test/test1/backup
INPUT_FILE=Benz11_noCT_S06B040US_N81k.def
RES_FILE=Benz11_withoutCT_S06B030_N81k_outM0.26_001.res
OUTPUT_TXT=perf-72.txt

cd ${WORK_DIR}

NP=$SLURM_PROCS
HOST_FILE=$(generate_pbs_nodefile)

cat ${HOST_FILE} > ${WORK_DIR}/HOST_STRING

HOST_STRING=""
for i_node in `cat ${HOST_FILE} | uniq`; do
	i_ppn=`cat ${HOST_FILE} | grep ${i_node} | wc -l`
	if [ -z ${HOST_STRING} ];then
		HOST_STRING="${i_node}*${i_ppn}"
	else
		HOST_STRING="${HOST_STRING},${i_node}*${i_ppn}"：
	fi
done

echo "The hosts is ${HOST_STRING}" >> ${WORK_DIR}/HOST_STRING

export CFX5RSH=ssh
${APP_EXEC} -par-dist ${HOST_STRING} -start-method 'Intel MPI Distributed Parallel' -def ${WORK_DIR}/${INPUT_FILE} -initial ${RES_FILE} 2>&1 | tee -a ${WORK_DIR}/file/${OUTPUT_TXT}
