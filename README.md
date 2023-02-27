# TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios

[**Webpage**](https://metadriverse.github.io/trafficgen/) | 
[**Code**](https://github.com/metadriverse/trafficgen) |
[**Video**](https://youtu.be/jPS93-d6msM) |
[**Paper**](https://arxiv.org/pdf/2210.06609.pdf)



## Setup environment

```bash
# Clone the code to local
git clone https://github.com/metadriverse/trafficgen.git
cd trafficgen

# Create virtual environment
conda create -n trafficgen python=3.8
conda activate trafficgen

# You should install pytorch by yourself to make them compatible with your GPU
# For cuda 11.0:
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# Install basic dependency
pip install -e .
```

If you find error messages related to `geos` when installing `Shapely`, checkout [this post](https://stackoverflow.com/questions/19742406/could-not-find-library-geos-c-or-load-any-of-its-variants).


## Quick Start

You can run the following scripts to test whether the setup is correct. These scripts do not require
downloading data.

#### Vehicle Placement Model
````
python train_init.py -c local
````
#### Trajectory Generator Model
````
python train_act.py -c local 
````


## Download and Process Dataset and Pre-trained Model

### Download dataset for road and traffic

Download from Waymo Dataset
- Register your Google account in: https://waymo.com/open/
- Open the following link with your Google account logged in: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0
- Download one or more proto files from `waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s`
- Move download files to `./raw_data/`

Note: You can download multiple files from above link and put them

### Data Preprocess

```bash
sh utils/data_trans.sh raw_data/dir processed_data/dir
```

[//]: # (The processed data has the following attributes:)

[//]: # (- `id`: scenario id)

[//]: # (- `all_agent`: A `[190, n, 9]` array which contains 190 frames, n agents, 9 features `[coord, velocity, heading, length, width, type, validity]`)

[//]: # (- `traffic_light`: A list containing information about the traffic light)

[//]: # (- `lane`: A `[n,4]` array which contains n points and `[coord, type, id&#40;which lane this point belongs to&#41;]` features.)

[//]: # ()

### Download and retrieve pretrained TrafficGen model

Please download two models from this link: https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

And then put them into `traffic_generator/ckpt` folder.

### Generate new traffic scenarios based on existing traffic scenarios

Running following scripts will generate images and GIFs (if with `--gif`) visualizing the new traffic scenarios in 
`traffic_generator/output/vis` folder.

```bash
# change the data usage and set the data dir in debug.yaml

# First, you have to change working directory
cd TrafficGen

python generate.py [--gif] 
```

Set `--gif` flag to generate GIF files.



## Training

### Train TrafficGen in the cluster
Modify cluster.yaml. Change the data path, data_usage.
````
python train_act.py -c cluster -d 0 1 2 3 -e test
````

-d denotes which GPU to use





