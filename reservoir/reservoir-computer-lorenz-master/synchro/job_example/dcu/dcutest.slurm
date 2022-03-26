#!/bin/bash
#SBATCH -o %j
#SBATCH -J OMP_DCU
#SBATCH -p huge
#SBATCH -t 00:30:00		#指定作业最大运行30分钟
#SBATCH --mem=90G		#占用节点全部内存
#SBATCH -N 4			#指定节点数
#SBATCH --ntasks-per-node=4	#指定每个节点的进程数
#SBATCH --ntasks-per-socket=1	#指定每个Socket的进程数，对应于NUMA node
#SBATCH --cpus-per-task=2	#指定每个进程的CPU数，对应多线程场景
#SBATCH --gres=dcu:4		#指定每个节点使用4块DCU卡

module load compiler/devtoolset/7.3.1
module load compiler/rocm/2.9
module load mpi/hpcx/2.4.1/gcc-7.3.1

# 处理OPENMP线程数变量
cpu_per_task=1
if [ -n "${SLURM_CPUS_PER_TASK}" ]; then
	cpu_per_task=$SLURM_CPUS_PER_TASK
fi
export OMP_NUM_THREADS=${cpu_per_task}

rocm-smi -a

env |grep SLURM

env |grep HIP

# 生成hostfile
hostfile=${SLURM_JOB_ID}.hostfile
for i in $(scontrol show hostnames $SLURM_NODELIST)
do
	# 需要指定--ntasks-per-node参数
	echo "$i slots=${SLURM_NTASKS_PER_NODE}"
done  > $hostfile

# 指定部分MPI参数
 export PARG=" -mca pml ucx -x UCX_IB_ADDR_TYPE=ib_global -x UCX_WARN_UNUSED_ENV_VARS=n -x LD_LIBRARY_PATH -mca btl_openib_warn_default_gid_prefix 0 -mca btl_openib_warn_no_device_params_found 0 -mca coll_hcoll_enable 0 -mca coll_hcoll_np ${SLURM_NTASKS} "

#指定部分MPI参数，按Socket映射绑定并且指定多线程参数
 export ARGS=" -np $SLURM_NTASKS --hostfile $hostfile --map-by socket:PE=${cpu_per_task} --bind-to core "

# 合并参数
export ARGS=" $ARGS $PARG "

echo "ARGS=$ARGS"

# 启动应用
 export APP="./xhpl 2"
 echo "use mpirun (numabind): " && time mpirun ${ARGS} ./numabind.sh ${APP}  


