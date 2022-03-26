#!/bin/bash
#SBATCH -J su2
#SBATCH -p debug
#SBATCH -N 10
#SBATCH -n 320
#SBATCH --ntasks-per-node=32
#SBATCH -o out.%j
#SBATCH -e err.%j

WDIR=`pwd`
cd $WDIR

##################################################################
#APP=/public/software/CAE/SU2_4.1.0/bin/run_SU2_CFD 
APP=/public/software/CAE/SU2_4.1.0/bin/SU2_CFD
INPUT_FILE=$WDIR/turb_SA_deltawing.cfg
##################################################################

NP=$SLURM_NPROCS
NNODE=`srun hostname |sort |uniq | wc -l`
LOG_FILE=$WDIR/job_${NP}c_${NNODE}n_$SLURM_JOB_ID.log
NODE_FILE=$WDIR/nodes_${NNODE}n_$SLURM_JOB_ID
HOST_FILE=$WDIR/hosts_${NP}c_${NNODE}n_$SLURM_JOB_ID
srun hostname | sort | uniq -c > $NODE_FILE
srun hostname | sort | uniq -c |awk '{printf "%s slots=%s \n",$2,$1}' > $HOST_FILE
#head -n $NNODE $ALLNODES > nodes
#sed "s/$/ slots=32/g" nodes > hosts

echo "Start time: `date +'%Y-%m-%d %H:%M:%S'`"  2>&1 | tee $LOG_FILE

cd $WDIR
mpirun --allow-run-as-root -np $NP -machinefile $HOST_FILE \
       -x UCX_LOG_LEVEL=ERROR \
       -mca pml ucx \
       --mca plm_rsh_no_tree_spawn 1 \
       --mca plm_rsh_num_concurrent $NNODE \
       -mca routed_radix $NNODE \
       -x UCX_IB_ADDR_TYPE=ib_global \
       -x UCX_DC_MLX5_TIMEOUT=35ms \
       -x UCX_RNDV_THRESH=16384 \
       -x UCX_ZCOPY_THRESH=16384 \
       -x UCX_MAX_EAGER_LANES=4 \
       -x UCX_MAX_RNDV_LANES=4 \
       -x LD_LIBRARY_PATH \
       -mca btl_openib_warn_default_gid_prefix 0 \
       -mca btl_openib_warn_no_device_params_found 0 \
       -mca coll_hcoll_enable 0 \
       -mca coll_hcoll_np $NP \
       --bind-to none \
       $APP $INPUT_FILE  2>&1 | tee -a $LOG_FILE

echo "End time: `date +'%Y-%m-%d %H:%M:%S'`" | tee -a $LOG_FILE
