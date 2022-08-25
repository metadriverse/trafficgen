#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python ../utils/trans20.py /data0/pengzh/training_20s/raw_${i} /data0/pengzh/v2_data_eval ${i} > ${i}.log 2>&1 &
done
