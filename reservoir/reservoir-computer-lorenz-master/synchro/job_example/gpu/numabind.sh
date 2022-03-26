#!/bin/bash

#APPCMD="/public/home/haowj/zhangtao/peixun_kunshan/open_fire_v8 ${LOOPMAX}"
APPCMD="$*"
echo "APPCMD=${APPCMD}"

lrank=$(expr $OMPI_COMM_WORLD_LOCAL_RANK % 4)
echo "[`hostname` PID=$$] OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK}, OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}, lrank=${lrank}"

export HIP_VISIBLE_DEVICES=${lrank}
export UCX_NET_DEVICES=mlx5_${lrank}:1
export UCX_IB_PCI_BW=mlx5_${lrank}:50Gbs
 numactl --cpunodebind=${lrank} --membind=${lrank} ${APPCMD}
