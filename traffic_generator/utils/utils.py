import argparse

def get_parsed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',type=str, default='local')
    parser.add_argument('--gif', action="store_true", help="How many scenarios you want to use.")
    args = parser.parse_args()
    return args