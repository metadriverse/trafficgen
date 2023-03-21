"""
This script provides a demo on how to create a single-agent MetaDrive environment with TrafficGen generated data.
The logic is simple: we replace the data folder of native Waymo dataset with the TrafficGen generated data.
Please read the definition of config, where some details about RL training are specified.

2023-03-21
PZH
"""

from metadrive.policy.replay_policy import ReplayEgoCarPolicy

import argparse

from metadrive.envs.real_data_envs.waymo_env import WaymoEnv

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
    parser.add_argument("--no_replay_traffic", action="store_true", help="If True, do not replay traffic vehicles' trajectories but instead use IDM policy to control all traffic vehicles.")
    parser.add_argument("--dataset", required=True, help=HELP)
    args = parser.parse_args()

    data_folder = os.path.join(root, args.dataset)
    assert os.path.isdir(data_folder), HELP

    config = dict(
        waymo_data_directory=data_folder,

        # MetaDrive will load pickle files with index [start_case_index, start_case_index + case_num)
        start_case_index=0,
        case_num=100,

        replay=False,
    )

    if args.replay:
        config["agent_policy"] = ReplayEgoCarPolicy

    env = WaymoEnv(config)

    for ep in tqdm.trange(100, desc="Episode"):
        env.reset()
        for t in tqdm.trange(1000, desc="Timestep"):
            o, r, d, i = env.step([0, 1])
            env.render(mode="topdown")
            if d:
                break