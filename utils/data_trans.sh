#!/usr/bin/env bash

#mkdir $1/raw_{0..9}
#
#for i in 0 1 2 3 4 5 6 7 8 9; do
#     mv $1/training_20s.tfrecord-00${i}* $1/raw_${i}/
#done

mkdir $1/trafficgen_data
for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python utils/trans20.py $1/raw_${i} $1/trafficgen_data ${i} > ${i}.log 2>&1 &
done
