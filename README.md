# TrafficGen: TODO: The title of the paper

TODO: some links here, take a look on MetaDrive project.

## Generating traffic flow with TrafficGen


### Step 1: Setup python environment

```bash
# Clone the code to local
git clone https://github.com/metadriverse/trafficgen.git
cd trafficgen

# Create virtual environment
conda create -n trafficgen python=3.7
conda activate trafficgen

# Install basic dependency
pip install -e .
```

### Step 2: Download dataset for road and traffic

Download from Waymo Dataset
- Register your Google account in: https://waymo.com/open/
- Open the following link with your Google account logged in: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0
- Download one or more proto files from `waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s`
- Move download files to `./raw_data/`

Note: You can download multiple files from above link and put them

### Step 3: Transform raw 
```bash
python trafficgen/scripts/trans20.py /raw_data /output_path None
```
The processed data has the following attributes

- `id`: scenario id
- `all_agent`: [190,n,9] 190 frames, n agents, 9 dim feature [coord,velocity,heading,length,width,type,validity]
- `traffic_light`
- `lane`: [n,4] n points, [coord,type,id(which lane this point belongs to)]

### Pretrained Model
https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

Put the pretrained model into ```/trafficgen/trafficen/model_weights```

## Generate Traffic Scenarios

```bash
# change the data usage and set the data dir in debug.yaml
vim cfg/debug.yaml

python generate_scenarios.py --cfg None
```