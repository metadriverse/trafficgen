#!/usr/bin/env bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python trans20.py /data/lanfeng/waymo/raw_${i} /data/lanfeng/waymo/init_data ${i} > ${i}.log 2>&1 &
done
