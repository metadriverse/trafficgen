## Installation

### Basic Installation

```bash
# Clone the code to local
git clone https://github.com/metadriverse/trafficgen.git
cd trafficgen

# Create virtual environment
conda create -n trafficgen python=3.7
conda activate trafficgen

# Install basic dependency
pip install -e .

cd trafficgen
```


### Waymo Dataset
- Register in https://waymo.com/open/
- Open https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false
- Download one proto file from 'waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s'

### Data Preprocess
```bash
python script/trans20.py /inp_path /output_path None
```
The processed data has the following attributes

- 'id': scenario id

- 'all_agent': [190,n,9] 190 frames, n agents, 9 dim feature [coord,velocity,heading,length,width,type,validity]

- 'traffic_light'

- 'lane': [n,4] n points, [coord,type,id(which lane this point belongs to)]

### Pretrained Model
https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

Put the pretrained model into ```/trafficgen/trafficen/model_weights```

## Generate Traffic Scenarios

```bash
# change the data usage and set the data dir in debug.yaml
vim cfg/debug.yaml

python generate_scenarios.py --cfg None
```