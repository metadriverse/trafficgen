# Waymo Dataset
https://waymo.com/open/

Download one proto file from
'waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s' for debug

Run "
python trans20.py /inp_path /outp_path None " to preprocess the proto file

The processed data can be used for training

processed data {

'id': scenario id

'all_agent': [190,n,9] 190 frames, n agents, 9 dim feature [coord,velocity,heading,length,width,type,validity]

'traffic_light'

'lane': [n,4] n points, [coord,type,id(which lane this point belongs to)]
}




# TrafficGen

0. pip install -e .

1. Download model to TrafficFormerV2/model_weights directory
https://drive.google.com/file/d/15mhhcolpVdg9oEphC6gxuODo9mtoWIBT/view?usp=sharing

2. python TrafficGen_demo.py --cfg debug


Generated traffic scenarios will be saved in TrafficFormerV2/heatmap
