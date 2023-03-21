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
- Move download files to PATH_A, where you store the raw tf_record files.

Note: it is not necessary to download all the files from Waymo. You can download one of them for a simple test.

Data Preprocess
```bash
python trafficgen/scripts/trans20.py PATH_A PATH_B None
```
Note: PATH_B is where you store the processed data.


[//]: # (The processed data has the following attributes:)

[//]: # (- `id`: scenario id)

[//]: # (- `all_agent`: A `[190, n, 9]` array which contains 190 frames, n agents, 9 features `[coord, velocity, heading, length, width, type, validity]`)

[//]: # (- `traffic_light`: A list containing information about the traffic light)

[//]: # (- `lane`: A `[n,4]` array which contains n points and `[coord, type, id&#40;which lane this point belongs to&#41;]` features.)

[//]: # ()

### Download and retrieve pretrained TrafficGen model

Please download two models from this link: https://drive.google.com/drive/folders/1TbCV6y-vssvG3YsuA6bAtD9lUX39DH9C?usp=sharing

And then put them into `trafficgen/traffic_generator/ckpt` folder.

### Generate new traffic scenarios

Running following scripts will generate images and GIFs (if with `--gif`) visualizing the new traffic scenarios in 
`traffic_generator/output/vis` folder.

```bash
# change the data usage and set the data dir in debug.yaml

# First, you have to change working directory
cd TrafficGen/trafficgen

python generate.py [--gif] [--save_metadrive]
```

Set `--gif` flag to generate GIF files.


## Connect TrafficGen with MetaDrive

After running `python generate.py --save_metadrive`,
a folder `trafficgen/traffic_generator/output/scene_pkl` will be created, and you will see many
pickle files. Each `.pkl` file is a scenario created by TrafficGen.

We provide a script to create single-agent RL environment with TrafficGen generated data.
Please refer to [trafficgen/run_metadrive.py](trafficgen/run_metadrive.py) for details.

We also provide pre-generated scenarios from TrafficGen, so you can kick off RL training
on TrafficGen-generated scenarios immediately. Please follow
[trafficgen/dataset/README.md](trafficgen/dataset/README.md)
to download the dataset.

```bash
cd trafficgen/

# Run generated scenarios:
python run_metadrive.py --dataset traffic_generator/output/scene_pkl

# Please read `trafficgen/dataset/README.md` to download pre-generated scenarios
# Then you can use them to create an RL environment:
python run_metadrive.py --dataset dataset/validation

# If you want to visualize the generated scenarios, with the ego car also replaying data, use:
python run_metadrive.py --dataset dataset/validation --replay

# If you want to create RL environment where traffic vehicles are not replaying 
# but are controlled by interactive IDM policy, use:
python run_metadrive.py --dataset dataset/validation --no_replay_traffic
```

You can then kick off RL training by utilizing the created environment showcased in the script above.


## Training

### Local Debug
Use the sample data packed in the code repo directly
#### Vehicle Placement Model
````
python train_init.py -c local
````
#### Trajectory Generator Model
````
python train_act.py -c local
````


### Cluster Training
For training, we recommend to download all the files from: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0

PATH_A is the raw data path

PATH_B is the processed data path

Execute the data_trans.sh:
```bash
sh utils/data_trans.sh PATH_A PATH_B
```
Note: This will take about 2 hours.

Then modify the 'data_path' in init/configs and act/configs to PATH_B, run:
```bash
python init/uitls/init_dataset.py
python act/uitls/act_dataset.py
```
to get a processed cache for the model.

Modify cluster.yaml. Change data_path, data_usage, run:
````
python train_act.py -c cluster -d 0 1 2 3 -e exp_name
````

-d denotes which GPU to use







