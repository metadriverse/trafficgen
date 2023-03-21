import argparse


def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='local')
    parser.add_argument('--gif', action="store_true", help="How many scenarios you want to use.")
    parser.add_argument('--save_metadrive', action="store_true",
                        help="Whether to save generated scenarios to MetaDrive-compatible file.")
    args = parser.parse_args()
    return args
