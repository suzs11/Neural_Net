#!/bin/bash

date
echo "Begin ..."
echo head node is `hostname`

source `pwd`/hpl.sh
mpirun --version
which mpirun

test_np=${1}
iter=${2}
hostfile=${3}

for i in $(seq 1 $iter)
do
mpirun --allow-run-as-root -np ${test_np} --hostfile `pwd`/${hostfile} -mca pml ucx -x UCX_IB_ADDR_TYPE=ib_global -x UCX_MAX_EAGER_LANES=4 -x UCX_MAX_RNDV_LANES=4 -x LD_LIBRARY_PATH -mca btl_openib_warn_default_gid_prefix 0 -mca btl_openib_warn_no_device_params_found 0 --bind-to none `pwd`/single_process.sh
sleep 4
done
echo "End ..."
date
