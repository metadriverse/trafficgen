import argparse

import torch


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='demo.yaml')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--exp_name', default="none", type=str)
    parser.add_argument('--num_scenarios', default=None, type=int, help="How many scenarios you want to use.")
    parser.add_argument('--gif', action="store_true", help="How many scenarios you want to use.")
    parser.add_argument('--save_pkl', action="store_true",
                        help="If you want to save the generated scenarios as pkl file")
    args = parser.parse_args()
    if args.exp_name.lower() == "none":
        args.exp_name = None
    args.rank = 0
    # when training on cluster, local rank need to be changed
    torch.cuda.set_device(args.local_rank)
    return args
