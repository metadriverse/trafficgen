import argparse
import os

import yaml

TRAFFICGEN_ROOT = os.path.dirname(os.path.dirname(__file__))

def load_config_act(path):
    """ load config file"""
    path = os.path.join('act/configs', f'{path}.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_config_init(path):
    """ load config file"""
    path = os.path.join(TRAFFICGEN_ROOT, "init", "configs", f'{path}.yaml')
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='cluster')
    parser.add_argument('--exp_name', '-e', default="test", type=str)
    parser.add_argument('--devices', '-d', nargs='+', default=[0, 1, 2, 3], type=int)
    args = parser.parse_args()
    return args
