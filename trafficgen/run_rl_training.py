from ray import tune
try:
    from metadrive.utils.waymo_utils.waymo_utils import AgentType
    from metadrive.utils.waymo_utils.waymo_utils import RoadEdgeType
    from metadrive.utils.waymo_utils.waymo_utils import RoadLineType
finally:
    pass
from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
from trafficgen.utils.training_utils import train, get_train_parser, DrivingCallbacks
from ray.rllib.agents.ppo import PPOTrainer

import os

# Path to: trafficgen/traffic_generator
root = os.path.join(os.path.abspath((os.path.dirname(__file__))))

HELP = \
"""
Please specify the path to data folder, which contains many pickle files.
For example, you can specify 'traffic_generator/output/scene_pkl' to load TrafficGen generated files.
You can also use 'dataset/generated_1385_training.zip' to specify pre-generated data. 
Please refer to 'dataset/README.md' for more information on pre-generated TrafficGen data.
"""


if __name__ == '__main__':
    parser = get_train_parser()
    parser.add_argument("--no_replay_traffic", action="store_true",
                        help="If True, do not replay traffic vehicles' trajectories but instead use IDM policy "
                             "to control all traffic vehicles.")
    parser.add_argument("--dataset_train", default="dataset/1385_training", help=HELP)
    parser.add_argument("--dataset_test", default="dataset/validation", help=HELP)
    parser.add_argument("--wandb", action="store_true", help="Whether to upload log to wandb.")
    parser.add_argument("--case_num_train", default=1385, type=int,
                        help="Number of scenarios you want to load from training dataset.")
    parser.add_argument("--case_num_test", default=100, type=int,
                        help="Number of scenarios you want to load from training dataset.")
    args = parser.parse_args()

    exp_name = args.exp_name or "TrafficGen_RL"
    stop = int(500_00000)
    case_num_train = args.case_num_train
    case_num_test = args.case_num_test
    replay_traffic = not args.no_replay_traffic

    data_folder_train = tune.grid_search([
        os.path.join(root, "dataset/1385_training"),
        os.path.join(root, "dataset/no_threshold_data"),
    ])
    # assert os.path.isdir(data_folder_train), "Can't find {}. ".format(data_folder_train) + HELP

    data_folder_test = os.path.join(root, args.dataset_test)
    assert os.path.isdir(data_folder_test), (
            "Can't find " + data_folder_test + ". It seems that you don't download the validation data. "
                                          "Please refer to 'dataset/README.md' for more information."
    )

    config = dict(

        # ===== Environment =====
        env=WaymoEnv,
        env_config=dict(
            waymo_data_directory=data_folder_train,

            # MetaDrive will load pickle files with index [start_case_index, start_case_index + case_num)
            start_case_index=0,
            case_num=case_num_train,

            replay=tune.grid_search([True, False]),
        ),

        # ===== Evaluation =====
        evaluation_interval=5,
        evaluation_num_episodes=40,
        evaluation_config=dict(env_config=dict(case_num=case_num_test, waymo_data_directory=data_folder_test)),
        evaluation_num_workers=2,
        metrics_smoothing_episodes=50,

        # ===== Training =====
        horizon=2000,
        num_sgd_iter=20,
        lr=3e-4,
        grad_clip=10.0,
        vf_clip_param=100.0,
        rollout_fragment_length="auto",
        sgd_minibatch_size=256,
        train_batch_size=30000,
        framework="torch",

        # ===== Resources =====
        num_gpus=0.2 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
        num_cpus_for_driver=0.5,
        num_workers=10,  # Number of parallel environments

    )

    kwargs = dict()
    if args.wandb:
        # Put your own wandb API token into the following file.
        kwargs["wandb_key_file"] = "~/wandb_api_key_file.txt"
        kwargs["wandb_project"] = "TrafficGen_RL"
        kwargs["wandb_team"] = "drivingforce"

    train(
        PPOTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        # num_seeds=args.num_seeds,
        num_seeds=1,
        custom_callback=DrivingCallbacks,
        test_mode=args.test,
        # local_mode=True

        # Put your wandb API to the following file, or do not call --wandb
        **kwargs
    )
