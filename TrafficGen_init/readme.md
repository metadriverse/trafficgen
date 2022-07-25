# Install

Create env

```conda create -n cvpr python=3.7```

Install drivingforce, in the root of drivingforce
```
pip install -e .
```

Install torch

```conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch```

```conda install yaml```

Install other dependencies

```pip install tqdm l5kit wandb matplotlib pandas seaborn tabulate```

# Local debug

download ```tt_debug.zip``` from google drive, uncompress it and put debug_data to TrafficTransformer

# Dataset Download

### install Google SDK
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
source ~/.bashrc



### Download Waymo Dataset (TFRecord)
## ERROR we use scenario 20 now !
gsutil -m cp -r "gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/" .

### Process Dataset
python dataTrans.py /rootpath
(if the downloaded dataset located in '/home/lfeng/waymo/tf_example', then rootpath is '/home/lfeng/waymo'. dataTrans will
generate processed data in '/home/lfeng/waymo/parsed')


# Tips About Training:
1. Per 10,000 data will consume 70 GB memory
2. DDP will load dataset for num_process times, while DP will only load one time
3. The true batch size is 16*cfg.batch_size*process_num, (for DP, process_num=1
4. Training scripts usage: ```bash train_distributed_ie.sh EXP_NAME``` (EXP_NAME is optional)
5. View log: ```tail -n 100 -f log.log```

# GPU-006 Usage

0. Password is the same as lfeng@IE SERVER
1. Connect jump server: ```ssh s1155136634@chpc-login01.itsc.cuhk.edu.hk```
2. After login, connect : ```ssh chpc-gpu006```
3. Download file: ```scp -r s1155136634@chpc-gpu006:/users/s1155136634/XYZxyz   /project/BoleiZhou/pengzh/drivingforce/results/PATH_IN_GATEWAY```
4. Down load file from jump server by FileZilla or ```scp -r s1155136634@chpc-login01.itsc.cuhk.edu.hk:/project/BoleiZhou/pengzh/drivingforce/results/PATH_IN_GATEWAY /Your/Local/Path/```

# Data process
TFRecord->dataTrans.py->wash_data->load_datset
