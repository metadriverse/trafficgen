nohup python translate_waymo_to_sd.py \
--input /data0/pengzh/waymo_1.2/training_20s \
--output /data0/pengzh/metadrive_processed_waymo/training_20s \
--num_workers 16 > 0427_translate.log 2>&1 &