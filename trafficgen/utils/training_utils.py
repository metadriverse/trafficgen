import argparse
import copy
import datetime
import json
import logging
import numbers
import os
import pickle
import time
from collections import defaultdict
from collections import deque
from collections.abc import Iterable
from multiprocessing import Queue
from typing import Dict
from typing import Optional

import numpy as np
import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.tune import CLIReporter

from multiprocessing import Queue

from ray.air.integrations.wandb import WandbLoggerCallback, _clean_log

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class OurWandbLogger(WandbLoggerCallback):
    def __init__(self, exp_name, *args, **kwargs):
        super(OurWandbLogger, self).__init__(*args, **kwargs)
        self.exp_name = exp_name

    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None

        # Project name for Wandb
        wandb_project = self.project

        # Grouping
        wandb_group = self.group or trial.experiment_dir_name if trial else None

        # remove unpickleable items!
        config = _clean_log(config)

        # ========== Our modification! ==========
        run_name = "{}_{}".format(self.exp_name, trial_id)

        wandb_init_kwargs = dict(
            id=trial_id,
            name=run_name,  # Our modification!
            resume=False,
            reinit=True,
            allow_val_change=True,
            group=wandb_group,
            project=wandb_project,
            config=config,
        )
        # ========== Our modification ends! ==========

        wandb_init_kwargs.update(self.kwargs)

        self._trial_queues[trial] = Queue()
        self._trial_processes[trial] = self._logger_process_cls(
            logdir=trial.logdir,
            queue=self._trial_queues[trial],
            exclude=exclude_results,
            to_config=self._config_results,
            **wandb_init_kwargs,
        )
        self._trial_processes[trial].start()


def get_api_key_file(wandb_key_file):
    if wandb_key_file is not None:
        default_path = os.path.expanduser(wandb_key_file)
    else:
        default_path = os.path.expanduser("~/wandb_api_key_file.txt")
    if os.path.exists(default_path):
        print("We are using this wandb key file: ", default_path)
        return default_path
    path = os.path.join(root, "wandb", "wandb_api_key_file.txt")
    print("We are using this wandb key file: ", path)
    return path


def train(
    trainer,
    config,
    stop,
    exp_name,
    num_seeds=1,
    num_gpus=0,
    test_mode=False,
    suffix="",
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    start_seed=0,
    local_mode=False,
    save_pkl=True,
    custom_callback=None,
    max_failures=0,
    # wandb support is removed!
    wandb_key_file=None,
    wandb_project=None,
    wandb_team=None,
    wandb_log_config=True,
    init_kws=None,
    **kwargs
):
    init_kws = init_kws or dict()
    # initialize ray
    if not os.environ.get("redis_password"):
        initialize_ray(test_mode=test_mode, local_mode=local_mode, num_gpus=num_gpus, **init_kws)
    else:
        password = os.environ.get("redis_password")
        assert os.environ.get("ip_head")
        print(
            "We detect redis_password ({}) exists in environment! So "
            "we will start a ray cluster!".format(password)
        )
        if num_gpus:
            print(
                "We are in cluster mode! So GPU specification is disable and"
                " should be done when submitting task to cluster! You are "
                "requiring {} GPU for each machine!".format(num_gpus)
            )
        initialize_ray(address=os.environ["ip_head"], test_mode=test_mode, redis_password=password, **init_kws)

    # prepare config
    used_config = {
        "seed": tune.grid_search([i * 100 + start_seed for i in range(num_seeds)]) if num_seeds is not None else None,
        "log_level": "DEBUG" if test_mode else "INFO",
        "callbacks": custom_callback if custom_callback else False,  # Must Have!
    }
    # if not custom_callback:
    #     used_config.pop("callbacks")
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    if isinstance(trainer, str):
        trainer_name = trainer
    elif hasattr(trainer, "_name"):
        trainer_name = trainer._name
    else:
        trainer_name = trainer.__name__

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if (keep_checkpoints_num is not None) and (not test_mode) and (keep_checkpoints_num != 0):
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    progress_reporter.add_metric_column("takeover")
    progress_reporter.add_metric_column("rc")
    kwargs["progress_reporter"] = progress_reporter

    if wandb_key_file is not None:
        assert wandb_project is not None
    if wandb_project is not None:
        assert wandb_project is not None
        kwargs["callbacks"] = [
            OurWandbLogger(
                exp_name=exp_name,
                api_key_file=get_api_key_file(wandb_key_file),
                project=wandb_project,
                group=exp_name,
                log_config=wandb_log_config,
                entity=wandb_team
            )
        ]

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=".",
        **kwargs
    )

    # save training progress as insurance
    if save_pkl:
        pkl_path = "{}-{}{}.pkl".format(exp_name, trainer_name, "" if not suffix else "-" + suffix)
        with open(pkl_path, "wb") as f:
            data = analysis.fetch_trial_dataframes()
            pickle.dump(data, f)
            print("Result is saved at: <{}>".format(pkl_path))
    return analysis


class ForceFPS:
    UNLIMITED = "UnlimitedFPS"
    FORCED = "ForceFPS"

    def __init__(self, fps, start=False):
        self.init_fps = fps
        if start:
            print("We will force the FPS to be near {}".format(fps))
            self.state = self.FORCED
            self.fps = fps + 1  # If we set to 10, FPS will jump in 9~10.
            self.interval = 1 / self.fps
        else:
            self.state = self.UNLIMITED
            self.fps = None
            self.interval = None
        self.dt_stack = deque(maxlen=10)
        self.last_time = time.time()

    def clear(self):
        self.dt_stack.clear()
        self.last_time = time.time()

    def sleep_if_needed(self):
        if self.fps is None:
            return
        self.dt_stack.append(time.time() - self.last_time)
        average_dt = sum(self.dt_stack) / len(self.dt_stack)
        if (self.interval - average_dt) > 0:
            time.sleep(self.interval - average_dt)
        self.last_time = time.time()


def merge_dicts(d1, d2):
    """
    Args:
        d1 (dict): Dict 1.
        d2 (dict): Dict 2.
    Returns:
         dict: A new dict that is d1 and d2 deep merged.
    """
    merged = copy.deepcopy(d1)
    deep_update(merged, d2, True, [])
    return merged


def deep_update(
    original, new_dict, new_keys_allowed=False, allow_new_subkey_list=None, override_all_if_type_changes=None
):
    """Updates original dict with values from new_dict recursively.
    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the allow_new_subkey_list, then new subkeys can be introduced.
    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        allow_new_subkey_list (Optional[List[str]]): List of keys that
            correspond to dict values where new subkeys can be introduced.
            This is only at the top level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), iff the "type" key in that value dict changes.
    """
    allow_new_subkey_list = allow_new_subkey_list or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}` ".format(k))

        # Both orginal value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Allowed key -> ok to add new subkeys.
            elif k in allow_new_subkey_list:
                deep_update(original[k], value, True)
            # Non-allowed key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M")


def same_padding(in_size, filter_size, stride_size):
    """
    PZH: Copied from RLLib.
    Note: Padding is added to match TF conv2d `same` padding. See
    www.tensorflow.org/versions/r0.12/api_docs/python/nn/convolution
    Args:
        in_size (tuple): Rows (Height), Column (Width) for input
        stride_size (Union[int,Tuple[int, int]]): Rows (Height), column (Width)
            for stride. If int, height == width.
        filter_size (tuple): Rows (Height), column (Width) for filter
    Returns:
        padding (tuple): For input into torch.nn.ZeroPad2d.
        output (tuple): Output shape after padding and convolution.
    """
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size

    stride_size = stride_size if isinstance(stride_size, Iterable) else [stride_size, stride_size]
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


class SafeJSONEncoder(json.JSONEncoder):
    def __init__(self, nan_str="null", **kwargs):
        super(SafeJSONEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if (type(value).__module__ == np.__name__ and isinstance(value, np.ndarray)):
                return value.tolist()

            if isinstance(value, np.bool_):
                return bool(value)

            if np.isnan(value):
                return self.nan_str

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeJSONEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False, **kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    if ray.__version__.split(".")[0] == "1":  # 1.0 version Ray
        if "redis_password" in kwargs:
            redis_password = kwargs.pop("redis_password")
            kwargs["_redis_password"] = redis_password

    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-gpus-per-trial", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    return parser


def setup_logger(debug=False):
    import logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["velocity"] = defaultdict(list)
        episode.user_data["steering"] = defaultdict(list)
        episode.user_data["step_reward"] = defaultdict(list)
        episode.user_data["acceleration"] = defaultdict(list)
        episode.user_data["cost"] = defaultdict(list)
        episode.user_data["episode_length"] = defaultdict(list)
        episode.user_data["episode_reward"] = defaultdict(list)
        episode.user_data["num_neighbours"] = defaultdict(list)
        episode.user_data["distance_error"] = defaultdict(list)
        # episode.user_data["distance_error_final"] = defaultdict(list)

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        active_keys = list(base_env.vector_env.envs[env_index].vehicles.keys())

        # The agent_rewards dict contains all agents' reward, not only the active agent!
        # active_keys = [k for k, _ in episode.agent_rewards.keys()]

        for agent_id in active_keys:
            k = agent_id
            info = episode.last_info_for(k)
            if info is None:
                info = episode.last_info_for()

            if info:
                if "step_reward" not in info:
                    continue
                episode.user_data["velocity"][k].append(info["velocity"])
                episode.user_data["steering"][k].append(info["steering"])
                episode.user_data["step_reward"][k].append(info["step_reward"])
                episode.user_data["acceleration"][k].append(info["acceleration"])
                episode.user_data["distance_error"][k].append(info["distance_error"])
                # episode.user_data["distance_error_final"][k].append(info["distance_error_final"])
                episode.user_data["cost"][k].append(info["cost"])
                episode.user_data["episode_length"][k].append(info["episode_length"])
                episode.user_data["episode_reward"][k].append(info["episode_reward"])
                episode.user_data["num_neighbours"][k].append(len(info.get("neighbours", [])))

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        keys = [k for k, _ in episode.agent_rewards.keys()]
        arrive_dest_list = []
        crash_list = []
        out_of_road_list = []
        max_step_rate_list = []

        # Newly introduced metrics
        track_length_list = []
        route_completion_list = []
        current_distance_list = []
        distance_error_final_list = []

        for k in keys:
            info = episode.last_info_for(k)
            if info is None:
                info = episode.last_info_for()

            arrive_dest = info.get("arrive_dest", False)

            # Newly introduced metrics
            route_completion = info.get("route_completion", None)
            assert route_completion is not None

            track_length = info.get("track_length", -1)
            current_distance = info.get("current_distance", -1)

            track_length_list.append(track_length)
            current_distance_list.append(current_distance)
            route_completion_list.append(route_completion)

            distance_error_final = info.get("distance_error_final", None)
            if distance_error_final is not None:
                distance_error_final_list.append(distance_error_final)

            crash = info.get("crash", False)
            out_of_road = info.get("out_of_road", False)
            max_step_rate = not (arrive_dest or crash or out_of_road)
            arrive_dest_list.append(arrive_dest)
            crash_list.append(crash)
            out_of_road_list.append(out_of_road)
            max_step_rate_list.append(max_step_rate)

        # Newly introduced metrics
        episode.custom_metrics["track_length"] = np.mean(track_length_list)
        episode.custom_metrics["current_distance"] = np.mean(current_distance_list)
        episode.custom_metrics["route_completion"] = np.mean(route_completion_list)
        episode.custom_metrics["distance_error_final"] = np.mean(distance_error_final_list)

        episode.custom_metrics["success_rate"] = np.mean(arrive_dest_list)
        episode.custom_metrics["crash_rate"] = np.mean(crash_list)
        episode.custom_metrics["out_of_road_rate"] = np.mean(out_of_road_list)
        episode.custom_metrics["max_step_rate"] = np.mean(max_step_rate_list)

        for info_k, info_dict in episode.user_data.items():
            self._add_item(episode, info_k, [vv for v in info_dict.values() for vv in v])

        agent_cost_list = [sum(episode_costs) for episode_costs in episode.user_data["cost"].values()]
        episode.custom_metrics["episode_cost"] = np.mean(agent_cost_list)
        episode.custom_metrics["episode_cost_worst_agent"] = np.min(agent_cost_list)
        episode.custom_metrics["episode_cost_best_agent"] = np.max(agent_cost_list)
        episode.custom_metrics["environment_cost_total"] = np.sum(agent_cost_list)
        episode.custom_metrics["num_active_agents"] = len(agent_cost_list)
        episode.custom_metrics["episode_length"] = np.mean(
            [ep_len[-1] for ep_len in episode.user_data["episode_length"].values()]
        )
        episode.custom_metrics["episode_reward"] = np.mean(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )
        episode.custom_metrics["environment_reward_total"] = np.sum(
            [ep_r[-1] for ep_r in episode.user_data["episode_reward"].values()]
        )

    def _add_item(self, episode, name, value_list):
        # episode.custom_metrics["{}_max".format(name)] = float(np.max(value_list))
        episode.custom_metrics["{}".format(name)] = float(np.mean(value_list))
        # episode.custom_metrics["{}_min".format(name)] = float(np.min(value_list))
        # pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["rc"] = np.nan
        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]

        if "route_completion_mean" in result["custom_metrics"]:
            result["rc"] = result["custom_metrics"]["route_completion_mean"]

        result["cost"] = np.nan
        if "episode_cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["episode_cost_mean"]

        # present the agent-averaged reward.
        result["raw_episode_reward_mean"] = result["episode_reward_mean"]

        # Fill Per agent reward as the item "episode_reward_mean", instead of the summation.
        policy_reward_mean = list(result["policy_reward_mean"].values())
        if len(policy_reward_mean) == 0:
            if "episode_reward_mean" in result["custom_metrics"]:
                result["episode_reward_mean"] = result["custom_metrics"]["episode_reward_mean"]
        else:
            result["episode_reward_mean"] = np.mean(policy_reward_mean)
        # result["environment_reward_total"] = np.sum(list(result["policy_reward_mean"].values()))
