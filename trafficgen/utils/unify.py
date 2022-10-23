from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    pass
import os


def unify_name(input_path):
    file_list = os.listdir(input_path)
    file_list.sort()
    for k, file_name in enumerate(tqdm(file_list)):
        file_path = os.path.join(input_path, file_name)
        new_file_path = os.path.join(input_path, "{}.pkl".format(k))
        os.rename(file_path, new_file_path)


if __name__ == "__main__":
    import sys

    raw_data_path = sys.argv[1]
    # raw_data_path = ".\\debug_data"
    unify_name(raw_data_path)
