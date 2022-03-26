#!/bim/bash

#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-5:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -p debug # 提交到哪一个分区
#SBATCH --mem=2000 # 所有核心可以使用的内存池大小，MB为单位
#SBATCH -o myjob.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e myjob.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=netid@nyu.edu # 把通知发送到哪一个邮箱
#SBATCH --constraint=2630v3  # the Features of the nodes, using command " showcluster " could find them.
#SBATCH --gres=gpu:n # 需要使用多少GPU，n是需要的数量""
