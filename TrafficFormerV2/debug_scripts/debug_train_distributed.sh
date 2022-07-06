#!/usr/bin/env bash

NOW="`date +%Y%m%d%H%M%S`"
NUM_PROC=4
MASTER_PORT=2333

if [ ! -d "output" ];then
    mkdir output
fi

python3 -m torch.distributed.launch --nproc_per_node=${NUM_PROC} --master_port=${MASTER_PORT} main.py --cfg="lqy_debug_remote" --distributed=True 2>&1 | tee output/tt_${NOW}.log
