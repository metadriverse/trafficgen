"""
This script provides a demo on how to create a single-agent MetaDrive environment with TrafficGen generated data.
The logic is simple: we replace the data folder of native Waymo dataset with the TrafficGen generated data.
Before running this script, please do this first:

1. Generate TrafficGen data by running the following command:
```bash
python generate.py --save_metadrive
```
2. Run this script to load the generated data and create a MetaDrive environment.


2024-05-23
PZH
"""

from metadrive.policy.replay_policy import ReplayEgoCarPolicy

import argparse

from metadrive.envs.scenario_env import ScenarioEnv

import os

import tqdm

# Path to: trafficgen/traffic_generator
root = os.path.join(os.path.abspath((os.path.dirname(__file__))))

HELP = \
"""
Please specify the path to data folder, which contains many pickle files.
For example, you can specify 'traffic_generator/output/scene_pkl' to load TrafficGen generated files.
You can also use 'dataset/generated_1385_training.zip' to specify pre-generated data. 
Please refer to 'dataset/README.md' for more information on pre-generated TrafficGen data.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true", help="Whether to replay ego vehicle actions.")
    parser.add_argument(
        "--no_replay_traffic",
        action="store_true",
        help=
        "If True, do not replay traffic vehicles' trajectories but instead use IDM policy to control all traffic vehicles."
    )
    # parser.add_argument("--dataset", default="dataset/validation", help=HELP)
    parser.add_argument("--dataset", default="traffic_generator/output/scene_pkl", help=HELP)
    args = parser.parse_args()

    data_folder = os.path.join(root, args.dataset)
    assert os.path.isdir(data_folder), "Can't find {}. ".format(data_folder) + HELP

    config = dict(
        data_directory=data_folder,

        # MetaDrive will load pickle files with index [start_case_index, start_case_index + case_num)
        start_scenario_index=0,
        num_scenarios=3,
        reactive_traffic=False,
    )

    if args.replay:
        config["agent_policy"] = ReplayEgoCarPolicy

    env = ScenarioEnv(config)

    for ep in tqdm.trange(100, desc="Episode"):
        env.reset()
        for t in tqdm.trange(1000, desc="Timestep"):
            o, r, d, _, i = env.step([0, 1])
            env.render(mode="topdown")
            if d:
                break
