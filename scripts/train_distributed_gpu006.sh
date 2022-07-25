#!/usr/bin/env bash
EXP_NAME=${1-none}
NOW="`date +%Y%m%d%H%M%S`"
NUM_PROC=8
MASTER_PORT=2333

if [ ! -d "output" ];then
    mkdir output
fi

python3 -m torch.distributed.launch --nproc_per_node=${NUM_PROC} --master_port=${MASTER_PORT} main.py --cfg="gpu006" --distributed=True --exp_name=${EXP_NAME} 2>&1 | tee output/tt_${NOW}.log
