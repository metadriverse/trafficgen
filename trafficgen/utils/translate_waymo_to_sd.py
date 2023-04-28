import argparse
import multiprocessing
import os

from metadrive.utils.waymo.script.convert_waymo_to_metadrive import parse_data
from tqdm.auto import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_FOLDER = os.path.join(REPO_ROOT, "trafficgen_v2", "data", "waymo")


def wrapped_parse_data(arg_list, input_path, output_path):
    assert len(arg_list) == 2
    return parse_data(arg_list[0], worker_index=arg_list[1], input_path=input_path, output_path=output_path)


def translate_waymo_to_sd(raw_data_path, output_path, num_workers=8):
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    file_list = sorted(os.listdir(raw_data_path))

    if num_workers > 1:
        from functools import partial
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        func = partial(wrapped_parse_data, input_path=raw_data_path, output_path=output_path)

        # func = lambda file_list, worker_index: parse_data(file_list, worker_index=worker_index, input_path=raw_data_path, output_path=output_path)

        # _wrapper = get_wrapper(input_path=raw_data_path, output_path=output_path)

        # Split the file list for each worker
        num_files_per_worker = int(len(file_list) // num_workers)
        file_list_per_worker = []
        for i in range(num_workers - 1):
            file_list_per_worker.append(file_list[i * num_files_per_worker:(i + 1) * num_files_per_worker])
        file_list_per_worker.append(file_list[(num_workers - 1) * num_files_per_worker:])
        assert sum(len(v) for v in file_list_per_worker) == len(file_list)
        assert len(file_list_per_worker) == num_workers

        argument_list = []
        for i in range(num_workers):
            argument_list.append([file_list_per_worker[i], i])

        with multiprocessing.Pool(num_workers) as p:
            return list(p.imap(func, argument_list))

    else:
        func = lambda worker_file_list: parse_data(worker_file_list, raw_data_path, output_path)
        return func(file_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="The data folder storing raw tfrecord from Waymo dataset.")
    parser.add_argument(
        "--output", default="processed_data", type=str, help="The data folder storing raw tfrecord from Waymo dataset."
    )
    parser.add_argument("--num_workers", default=0, type=int, help="The number of parallel workers.")
    args = parser.parse_args()

    scenario_data_path = args.input
    num_workers = args.num_workers

    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    raw_data_path = scenario_data_path

    translate_waymo_to_sd(raw_data_path, output_path, num_workers=num_workers)
