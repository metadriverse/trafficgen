# Waymo Dataset
https://waymo.com/open/

Download one proto file from
'waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s' for debug

Run "
python script/trans20.py /inp_path /outp_path None " to preprocess the proto file

processed data {

'id': scenario id

'all_agent': [190,n,9] 190 frames, n agents, 9 dim feature [coord,velocity,heading,length,width,type,validity]

'traffic_light'

'lane': [n,4] n points, [coord,type,id(which lane this point belongs to)]
}




# TrafficGen

0. pip install -e .

1. Download model to trafficgen/model_weights directory
https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

2. python generate_scenarios.py --cfg debug

