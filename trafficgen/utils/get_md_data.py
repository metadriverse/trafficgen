import argparse
import os
import pickle

import numpy as np
from metadrive.scenario import ScenarioDescription as SD, MetaDriveType
from metadrive.utils.waymo_utils.utils import read_waymo_data
from tqdm import tqdm

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
    'SPEED_BUMP': 19,
}

SAMPLE_NUM = 10


def down_sampling(line):
    # if is center lane
    point_num = line.shape[0]

    ret = []

    if point_num < SAMPLE_NUM:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret


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
        all_agent[:, indx, 5:7] = track[SD.STATE]['size'][:, :2]
        all_agent[:, indx, 7] = 1
        all_agent[:, indx, 8] = track[SD.STATE]['valid'].reshape(track_len)

    assert sdc_index is not None

    # Make ego agent to the first place
    all_agent[:, [sdc_index, 0]] = all_agent[:, [0, sdc_index]]

    ret['all_agent'] = all_agent

    traffic_light_data = []
    for step in range(track_len):
        # t_l_t = []
        # tls_t = traffic_lights[i]

        tl_states_in_one_step = []

        for traffic_light_index, traffic_light in traffic_lights.items():
            traffic_light_state = {k: v[step] for k, v in traffic_light["state"].items()}

            traffic_light_step_data = np.zeros(6, dtype='float32')

            # TODO: The range of this data is int [0, 253]. I don't think it's meaningful reading this data.
            traffic_light_step_data[0] = traffic_light_state["lane"]

            # TODO: The range of this data is float with shape [200, 3] in range [-352, 169].
            traffic_light_step_data[1:3] = traffic_light_state["stop_point"][:2]

            # Int in range [0, 3], stands for UNKNOWN, STOP, CAUTION, GO
            traffic_light_step_data[4] = traffic_light_state["object_state"]

            # Whether valid
            traffic_light_step_data[5] = 1 if traffic_light_state["object_state"] else 0

            tl_states_in_one_step.append(traffic_light_step_data)

        traffic_light_data.append(tl_states_in_one_step)

        # for j in range(len(tls_t)):
        #     t_l_t_j = np.zeros(6, dtype='float32')
        #     t_l_t_j[0] = tls_t[j]['lane']
        #     t_l_t_j[1:3] = tls_t[j]['stop_point'][:2]
        #     state = tls_t[j]['state']
        #     if 'GO' in state:
        #         t_l_t_j[4] = 3
        #     elif 'CAUTION' in state:
        #         t_l_t_j[4] = 2
        #     elif 'STOP' in state:
        #         t_l_t_j[4] = 1
        #     else:
        #         t_l_t_j[4] = 0
        #     t_l_t_j[5] = 1 if t_l_t_j[4] else 0
        #     t_l_t.append(t_l_t_j)
        # traffic_light.append(t_l_t)

    ret['traffic_light'] = traffic_light_data

    lanes = []
    for map_feat_id, map_feat in map_feat.items():

        if "polyline" not in map_feat:
            map_feat['polyline'] = map_feat['position'][np.newaxis]

        poly_unsampled = map_feat['polyline'][:, :2]

        # TODO(PZH): Revisit the down sampling function. It seems quite werid to me.
        poly = down_sampling(poly_unsampled)

        a_lane = np.zeros([len(poly), 4], dtype='float32')

        a_lane[:, :2] = np.array(poly)
        a_lane[:, 2] = ALL_TYPE[map_feat['type']]
        a_lane[:, 3] = 1

        lanes.append(a_lane)
    lanes = np.concatenate(lanes, axis=0)

    ret['lane'] = lanes

    return ret


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

    cnt = 0

    batch = []

    for pickle_file in tqdm(pickle_files):
        md_path = os.path.join(input_folder, pickle_file)
        scenario = read_waymo_data(md_path)
        transformed = metadrive_scenario_to_init_data(scenario)

        # TODO(PZH): Temporarily remove post-processing. Decide later!
        batch.append(transformed)

        out_path = os.path.join(output_folder, "{}.pkl".format(cnt))
        with open(out_path, "wb") as f:
            pickle.dump(transformed, f)
            print("File is saved at: ", out_path)
        cnt += 1
