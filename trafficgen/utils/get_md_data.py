"""
PZH Note:

This file provides a function to read latest MetaDrive Scenario Description (SD)
to the internal data representation of TrafficGen.

The test script provided below allows you to read the pickle files storing SD
and directly connect it to TrafficGen placing vehicles functionality.

The output images after placing vehicles in the scenes will be saved to the TMP_IMG folder.
"""
import argparse
import os
import pickle

import numpy as np
import torch
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary
from tqdm import tqdm
from metadrive.type import MetaDriveType
import pathlib

ALL_TYPE = {
    'LANE_FREEWAY': 1,
    'LANE_SURFACE_STREET': 2,
    'LANE_BIKE_LANE': 3,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 6,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 7,
    'ROAD_LINE_SOLID_DOUBLE_WHITE': 8,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW': 9,
    'ROAD_LINE_BROKEN_DOUBLE_YELLOW': 10,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 11,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW': 12,
    'ROAD_LINE_PASSING_DOUBLE_YELLOW': 13,
    'ROAD_EDGE_BOUNDARY': 15,
    'ROAD_EDGE_MEDIAN': 16,
    'STOP_SIGN': 17,

    'CROSS_WALK': 18,
    'CROSSWALK': 18,

    'SPEED_BUMP': 19,

    # This is newly introduced in WOMD 1.2.0 (TODO: Not sure if setting it to 20 is bug-free)
    'DRIVEWAY': 20,
}

METADRIVE_TYPE_TO_INT = ALL_TYPE
INT_TO_METADRIVE_TYPE = {v: k for k, v in METADRIVE_TYPE_TO_INT.items()}


def _down_sampling(line, sample_num):
    # if is center lane
    point_num = line.shape[0]

    ret = []

    if point_num < sample_num:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, sample_num):
            ret.append(line[i])

    return ret


class RoadLineType(Enum):
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @classmethod
    def lookup(cls, v):
        for k, vv in cls.__dict__.items():
            if vv == v:
                return k
        return "UNKNOWN"


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k]["lane_start_index"], fb[k]["lane_end_index"]]
        c['type'] = RoadLineType.lookup(fb[k]["boundary_type"])
        c['id'] = fb[k]["boundary_feature_id"]
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k]["feature_id"]
        nb['indexes'] = [
            fb[k]["self_start_index"], fb[k]["self_end_index"], fb[k]["neighbor_start_index"], fb[k]["neighbor_end_index"]
        ]
        nb['indexes'] = [
            fb[k]["self_start_index"], fb[k]["self_end_index"], fb[k]["neighbor_start_index"], fb[k]["neighbor_end_index"]
        ]
        nb['boundaries'] = extract_boundaries(fb[k]["boundaries"])
        nb['id'] = fb[k]["feature_id"]
        nbs.append(nb)
    return nbs


def extract_center(lane):
    center = {}
    # f = f.lane

    def down_sampling(line, type=0):
        # if is center lane
        point_num = len(line)

        ret = []
        SAMPLE_NUM = 10
        if point_num < SAMPLE_NUM or type == 1:
            for i in range(0, point_num):
                ret.append(line[i])
        else:
            for i in range(0, point_num, SAMPLE_NUM):
                ret.append(line[i])

        return ret


    poly = down_sampling(lane['polyline'][:, :2])

    t = METADRIVE_TYPE_TO_INT[lane["type"]]
    poly = [np.insert(x, 2, t) for x in poly]

    center['interpolating'] = lane["interpolating"]

    center['entry'] = [x for x in lane["entry_lanes"]]

    center['exit'] = [x for x in lane["exit_lanes"]]

    center['left_boundaries'] = extract_boundaries(lane["left_boundaries"])

    center['right_boundaries'] = extract_boundaries(lane["right_boundaries"])

    center['left_neighbor'] = extract_neighbors(lane["left_neighbor"])

    center['right_neighbor'] = extract_neighbors(lane["right_neighbor"])

    return poly, center




def _extract_map(map_feats, sample_num):
    lanes = []

    center_infos = {}

    for map_feat_id, map_feat in map_feats.items():

        if map_feat["type"] == "DRIVEWAY":
            # TODO: The driveway is not supported in the current version of TrafficGen.
            continue

        if "polyline" not in map_feat:
            if "polygon" in map_feat:
                map_feat['polyline'] = map_feat['polygon'] #[np.newaxis]
            else:
                map_feat['polyline'] = map_feat['position'][np.newaxis]

        poly_unsampled = map_feat['polyline'][:, :2]

        # TODO(PZH): Revisit the down sampling function. It seems quite werid to me.
        poly = _down_sampling(poly_unsampled, sample_num=sample_num)

        a_lane = np.zeros([len(poly), 4], dtype='float32')

        a_lane[:, :2] = np.array(poly)
        a_lane[:, 2] = METADRIVE_TYPE_TO_INT[map_feat['type']]
        a_lane[:, 3] = str(map_feat_id)

        lanes.append(a_lane)

        if "LANE" in map_feat["type"]:
            center_info = extract_center(map_feat)
            center_infos[str(map_feat_id)] = center_info



    lanes = np.concatenate(lanes, axis=0)
    return lanes, center_info


def metadrive_scenario_to_init_data(scenario):
    ret = {}

    ret['id'] = scenario[SD.ID]

    tracks = scenario[SD.TRACKS]
    traffic_lights = scenario[SD.DYNAMIC_MAP_STATES]
    map_feat = scenario[SD.MAP_FEATURES]
    sdc_id = scenario[SD.METADATA][SD.SDC_ID]

    track_len = scenario[SD.LENGTH]

    # all_agent in shape [Time steps, Num agents, Num state dim]
    all_agent = np.zeros([track_len, len(tracks), 9], dtype='float32')

    sdc_index = None
    for indx, (id, track) in enumerate(tracks.items()):
        if track[SD.TYPE] != MetaDriveType.VEHICLE:
            continue
        if id == sdc_id:
            sdc_index = indx
        all_agent[:, indx, :2] = track[SD.STATE]['position'][:, :2]
        all_agent[:, indx, 2:4] = track[SD.STATE]['velocity']
        all_agent[:, indx, 4] = track[SD.STATE]['heading'].reshape(track_len)
        all_agent[:, indx, 5] = track[SD.STATE]['length']  # [:, :2]
        all_agent[:, indx, 6] = track[SD.STATE]['width']  # [:, :2]
        all_agent[:, indx, 7] = 1
        all_agent[:, indx, 8] = track[SD.STATE]['valid'].reshape(track_len)

    assert sdc_index is not None

    # Make ego agent to the first place
    all_agent[:, [sdc_index, 0]] = all_agent[:, [0, sdc_index]]

    ret['all_agent'] = all_agent

    traffic_light_data = []
    for step in range(track_len):
        tl_states_in_one_step = []

        for traffic_light_index, traffic_light in traffic_lights.items():
            traffic_light_state = {k: v[step] for k, v in traffic_light["state"].items()}

            traffic_light_step_data = np.zeros(6, dtype='float32')

            # The range of this data is int [0, 253]. Will use to filter lanes. It is very useful.
            traffic_light_step_data[0] = str(traffic_light["lane"])

            # TODO: The range of this data is float with shape [200, 3] in range [-352, 169].
            traffic_light_step_data[1:3] = traffic_light["stop_point"][:2]

            # Int in range [0, 3], stands for UNKNOWN, STOP, CAUTION, GO

            s = MetaDriveType.simplify_light_status(traffic_light_state["object_state"] )
            if s == MetaDriveType.LIGHT_RED:
                traffic_light_step_data[4] = 1
            elif s == MetaDriveType.LIGHT_YELLOW:
                traffic_light_step_data[4] = 2
            elif s == MetaDriveType.LIGHT_GREEN:
                traffic_light_step_data[4] = 3
            else:
                traffic_light_step_data[4] = 0

            # Whether valid
            traffic_light_step_data[5] = not (traffic_light_state["object_state"] == MetaDriveType.LANE_STATE_UNKNOWN)

            tl_states_in_one_step.append(traffic_light_step_data)

        traffic_light_data.append(tl_states_in_one_step)

    ret['traffic_light'] = traffic_light_data

    ret['lane'], ret['center_info'] = _extract_map(map_feat, sample_num=10)
    ret['unsampled_lane'], _ = _extract_map(map_feat, sample_num=10e9)

    # ret["original_metadrive_scenario"] = scenario

    return ret


def extend_batch_dim(data):
    new_data = {}
    for k, tensor in data.items():
        if k != "other":
            new_data[k] = np.expand_dims(tensor, 0)

    new_data["other"] = {}
    for k, tensor in data["other"].items():
        if k == "traf":  # What the fuck this name is?
            pass
        else:
            tensor = np.expand_dims(tensor, 0)
        new_data["other"][k] = tensor

    return new_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="raw_data", help="The folder of input data.")
    parser.add_argument("--output", default="test_output", help="The folder of output data.")
    parser.add_argument("--num_scenarios", "-n", default=-1, type=int)  # -1 stands for loading all
    parser.add_argument('--config', '-c', type=str, default='local')
    args = parser.parse_args()

    input_folder = args.input
    assert os.path.isdir(input_folder)


    # pickle_files = [p for p in os.listdir(input_folder) if p.endswith(".pkl")]
    _, _, pickle_files = read_dataset_summary(input_folder)

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    num_scenarios = args.num_scenarios
    if num_scenarios == -1:
        num_scenarios = len(pickle_files)

    vis_dir = "TMP_IMG"
    os.makedirs(vis_dir, exist_ok=True)

    cnt = 0

    from trafficgen.traffic_generator.traffic_generator import TrafficGen
    from trafficgen.utils.config import load_config_init
    from trafficgen.traffic_generator.utils.data_utils import process_data_to_internal_format

    cfg = load_config_init(args.config)
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrafficGen(cfg)

    batch = []

    for index, (pickle_file, path) in enumerate(tqdm(pickle_files.items())):
        md_path = os.path.join(input_folder, path, pickle_file)
        scenario = read_scenario_data(md_path)
        transformed = metadrive_scenario_to_init_data(scenario)

        batch.append(transformed)

        out_path = pathlib.Path(output_folder) / "{}.pkl".format(cnt)
        out_path = out_path.resolve()
        with open(out_path, "wb") as f:
            pickle.dump(transformed, f)
            print("The TrafficGen's internal data is saved at: ", out_path)
        cnt += 1


        print("Visualizing the placed vehicles in the scenario... (Comment this out if you want faster convertion.")
        internal_data = process_data_to_internal_format(transformed)
        data = internal_data[0]
        data = extend_batch_dim(data)
        for k in data:
            if isinstance(data[k], torch.Tensor) and (data[k].device != model.act_model.device):
                data[k] = data[k].to(model.act_model.device)
        model.place_vehicles_for_single_scenario(data, index=index, vis=True, vis_dir=vis_dir)
