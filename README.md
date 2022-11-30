# TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios

[**Webpage**](https://metadriverse.github.io/trafficgen/) | 
[**Code**](https://github.com/metadriverse/trafficgen) |
[**Video**](https://youtu.be/jPS93-d6msM) |
[**Paper**](https://arxiv.org/pdf/2210.06609.pdf)


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

### Step 3: Transform raw data in TF files to python objects

```bash
python trafficgen/scripts/trans20.py raw_data processed_data None
```

The processed data has the following attributes:
- `id`: scenario id
- `all_agent`: A `[190, n, 9]` array which contains 190 frames, n agents, 9 features `[coord, velocity, heading, length, width, type, validity]`
- `traffic_light`: A list containing information about the traffic light
- `lane`: A `[n,4]` array which contains n points and `[coord, type, id(which lane this point belongs to)]` features.

### Step 4: Download and retrieve pretrained TrafficGen model

Please download two models from this link: https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

And then put them into `./trafficgen/model_weights` folder.

### Step 5: Generate new traffic scenarios based on existing traffic scenarios

Running following scripts will generate images and GIFs (if with `--gif`) visualizing the new traffic scenarios in 
`./vis` folder.

```bash
# change the data usage and set the data dir in debug.yaml

# First, you have to change working directory
cd trafficgen

python trafficgen/generate_scenarios.py [--gif] [--num_scenarios 10]
```

Set `--gif` flag to generate GIF files.


