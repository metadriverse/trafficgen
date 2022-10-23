from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    pass
import os
import pickle


def unify_name(input_path):
    for k, file_name in enumerate(tqdm(file_list)):
        file_path = os.path.join(input_path, file_name)
        new_file_path = os.path.join(input_path, "{}.pkl".format(k))
        os.rename(file_path, new_file_path)


if __name__ == "__main__":
    import sys

    raw_data_path = sys.argv[1]
    file_list = os.listdir(raw_data_path)
    file_list.sort()
    with open('./name.pkl', 'wb') as f:
        pickle.dump(file_list, f)
