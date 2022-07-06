#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
    nohup python ../trans20.py /data0/pengzh/scene_v110/raw_${i} /data0/pengzh/v2_data_metadrive ${i} > ${i}.log 2>&1 &
done
