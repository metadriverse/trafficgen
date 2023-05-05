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

import numpy as np
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.utils.waymo_utils.utils import read_waymo_data
from tqdm import tqdm

polyline_type_sd_to_waymo = {
    # for lane
    MetaDriveType.LANE_UNKNOWN: -1,
    MetaDriveType.LANE_FREEWAY: 1,
    MetaDriveType.LANE_SURFACE_STREET: 2,
    MetaDriveType.LANE_BIKE_LANE: 3,

    # for roadline
    MetaDriveType.LINE_UNKNOWN: -1,
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 6,
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: 7,
    MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 8,
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 9,
    MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 10,
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 11,
    MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 12,
    MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 13,

    # for roadedge
    MetaDriveType.BOUNDARY_UNKNOWN: -1,
    MetaDriveType.BOUNDARY_LINE: 15,
    MetaDriveType.BOUNDARY_MEDIAN: 16,

    # for stopsign
    MetaDriveType.STOP_SIGN: 17,

    # for crosswalk
    MetaDriveType.CROSSWALK: 18,

    # for speed bump
    MetaDriveType.SPEED_BUMP: 19,

    # driveway
    MetaDriveType.DRIVEWAY: 20,
}

traffic_light_state_to_int = {
    None: 0,
    'LANE_STATE_UNKNOWN': 0,

    # // States for traffic signals with arrows.
    'LANE_STATE_ARROW_STOP': 1,
    'LANE_STATE_ARROW_CAUTION': 2,
    'LANE_STATE_ARROW_GO': 3,

    # // Standard round traffic signals.
    'LANE_STATE_STOP': 4,
    'LANE_STATE_CAUTION': 5,
    'LANE_STATE_GO': 6,

    # // Flashing light signals.
    'LANE_STATE_FLASHING_STOP': 7,
    'LANE_STATE_FLASHING_CAUTION': 8,
}



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

def _extract_map(map_feat, sample_num):
    lanes = []

    center_info = []

    for map_feat_id, map_feat in map_feat.items():

        if "polyline" not in map_feat:
            if "position" in map_feat:
                pos = map_feat['position'][np.newaxis, :2]
                # Repeat the polyline to avoid it being ignored in vectorization.
                map_feat['polyline'] = pos.repeat(3, 1)
            elif "polygon" in map_feat:
                map_feat['polyline'] = map_feat['polygon'][:, :2]
            else:
                raise ValueError()

        poly_unsampled = map_feat['polyline'][:, :2]

        # TODO(PZH): Revisit the down sampling function. It seems quite werid to me.
        poly = _down_sampling(poly_unsampled, sample_num=sample_num)

        a_lane = np.zeros([len(poly), 4], dtype='float32')

        a_lane[:, :2] = np.array(poly)
        a_lane[:, 2] = polyline_type_sd_to_waymo[map_feat['type']]
        a_lane[:, 3] = str(map_feat_id)

        lanes.append(a_lane)

    lanes = np.concatenate(lanes, axis=0)
    return lanes



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
        all_agent[:, indx, 5] = track[SD.STATE]["length"]
        all_agent[:, indx, 6] = track[SD.STATE]["width"]
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
            traffic_light_step_data[4] = traffic_light_state_to_int[traffic_light_state["object_state"]]

            # Whether valid
            traffic_light_step_data[5] = 1 if traffic_light_state["object_state"] else 0

            tl_states_in_one_step.append(traffic_light_step_data)

        traffic_light_data.append(tl_states_in_one_step)

    ret['traffic_light'] = traffic_light_data

    ret['lane'] = _extract_map(map_feat, sample_num=10)
    ret['unsampled_lane'] = _extract_map(map_feat, sample_num=10e9)

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
    args = parser.parse_args()

    input_folder = args.input
    assert os.path.isdir(input_folder)
    pickle_files = [p for p in os.listdir(input_folder) if p.endswith(".pkl")]

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)

    num_scenarios = args.num_scenarios
    if num_scenarios == -1:
        num_scenarios = len(pickle_files)

    vis_dir = "TMP_IMG"
    os.makedirs(vis_dir, exist_ok=True)

    cnt = 0

    from trafficgen.traffic_generator.traffic_generator import TrafficGen
    from trafficgen.traffic_generator.utils.utils import get_parsed_args
    from trafficgen.utils.config import load_config_init
    from trafficgen.traffic_generator.utils.data_utils import process_data_to_internal_format

    args = get_parsed_args()
    cfg = load_config_init(args.config)
    model = TrafficGen(cfg)

    batch = []

    for index, pickle_file in enumerate(tqdm(pickle_files)):
        md_path = os.path.join(input_folder, pickle_file)
        scenario = read_waymo_data(md_path)
        transformed = metadrive_scenario_to_init_data(scenario)

        # TODO(PZH): Temporarily remove post-processing. Decide later!
        batch.append(transformed)

        # out_path = os.path.join(output_folder, "{}.pkl".format(cnt))
        # with open(out_path, "wb") as f:
        #     pickle.dump(transformed, f)
        #     print("File is saved at: ", out_path)
        # cnt += 1

        internal_data = process_data_to_internal_format(transformed)
        data = internal_data[0]
        data = extend_batch_dim(data)
        model.place_vehicles_for_single_scenario(data, index=index, vis=True, vis_dir=vis_dir)
