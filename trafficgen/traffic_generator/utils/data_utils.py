import copy
import os
import pickle

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType
from trafficgen.utils.utils import process_map, rotate, cal_rel_dir, WaymoAgent


TRAFFICGEN_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

LANE_SAMPLE = 10
RANGE = 50


def process_case_to_input(case, agent_range=60):
    inp = {}
    agent = WaymoAgent(case['agent'])
    agent = agent.get_inp(act_inp=True)
    range_mask = (abs(agent[:, 0] - 40 / 50) < agent_range / 50) * (abs(agent[:, 1]) < agent_range / 50)
    agent = agent[range_mask]

    agent = agent[:32]
    mask = np.ones(agent.shape[0])
    mask = mask[:32]
    agent = np.pad(agent, ([0, 32 - agent.shape[0]], [0, 0]))
    mask = np.pad(mask, ([0, 32 - mask.shape[0]]))
    inp['agent'] = agent
    inp['agent_mask'] = mask.astype(bool)

    inp['center'], inp['center_mask'], inp['bound'], inp['bound_mask'], \
    inp['cross'], inp['cross_mask'], inp['rest'], inp['rest_mask'] = process_map(
        case['lane'][np.newaxis], [case['traf']], center_num=256, edge_num=128, offest=-40, lane_range=60)

    inp['center'] = inp['center'][0]
    inp['center_mask'] = inp['center_mask'][0]
    inp['bound'] = inp['bound'][0]
    inp['bound_mask'] = inp['bound_mask'][0]
    inp['cross'] = inp['cross'][0]
    inp['cross_mask'] = inp['cross_mask'][0]
    inp['rest'] = inp['rest'][0]
    inp['rest_mask'] = inp['rest_mask'][0]
    return inp


def get_type_class(line_type):
    from metadrive.scenario import ScenarioDescription as SD, MetaDriveType
    from metadrive.constants import LineType

    if line_type in range(1, 4):
        return MetaDriveType.LANE_CENTER_LINE
    elif line_type == 6:
        return MetaDriveType.BROKEN_GREY_LINE
        # return RoadLineType.BROKEN_SINGLE_WHITE
    elif line_type == 7:
        return MetaDriveType.CONTINUOUS_GREY_LINE
        # return RoadLineType.SOLID_SINGLE_WHITE
    elif line_type == 8:
        return MetaDriveType.CONTINUOUS_GREY_LINE
        # return RoadLineType.SOLID_DOUBLE_WHITE
    elif line_type == 9:
        return MetaDriveType.BROKEN_YELLOW_LINE
        # return RoadLineType.BROKEN_SINGLE_YELLOW
    elif line_type == 10:
        return MetaDriveType.BROKEN_YELLOW_LINE
        # return RoadLineType.BROKEN_DOUBLE_YELLOW
    elif line_type == 11:
        return MetaDriveType.CONTINUOUS_YELLOW_LINE
        # return RoadLineType.SOLID_SINGLE_YELLOW
    elif line_type == 12:
        return MetaDriveType.CONTINUOUS_YELLOW_LINE
        # return RoadLineType.SOLID_DOUBLE_YELLOW
    elif line_type == 13:
        # return RoadLineType.PASSING_DOUBLE_YELLOW
        return MetaDriveType.BROKEN_YELLOW_LINE
    elif line_type == 15:
        # return RoadEdgeType.BOUNDARY
        return LineType.SIDE
    elif line_type == 16:
        # return RoadEdgeType.MEDIAN
        return LineType.SIDE
    else:
        return MetaDriveType.UNKNOWN_LINE
        # return 'other'


def from_list_to_batch(inp_list):
    keys = inp_list[0].keys()

    batch = {}
    for key in keys:
        one_item = [item[key] for item in inp_list]
        batch[key] = Tensor(np.stack(one_item))

    return batch


def transform_to_agent(agent_i, agent, lane):
    all_ = copy.deepcopy(agent)

    center = copy.deepcopy(agent_i[:2])
    center_yaw = copy.deepcopy(agent_i[4])

    all_[..., :2] -= center
    coord = rotate(all_[..., 0], all_[..., 1], -center_yaw)
    vel = rotate(all_[..., 2], all_[..., 3], -center_yaw)

    all_[..., :2] = coord
    all_[..., 2:4] = vel
    all_[..., 4] = all_[..., 4] - center_yaw
    # then recover lane's position
    lane = copy.deepcopy(lane)
    lane[..., :2] -= center
    output_coords = rotate(lane[..., 0], lane[..., 1], -center_yaw)
    if isinstance(lane, Tensor):
        output_coords = Tensor(output_coords)
    lane[..., :2] = output_coords

    return all_, lane

def _traffic_light_state_template(object_id, track_length):
    """Borrowed from MetaDrive"""
    from metadrive.scenario import ScenarioDescription as SD, MetaDriveType
    return dict(
        type=MetaDriveType.TRAFFIC_LIGHT,
        state=dict(
            stop_point=np.zeros([track_length, 3], dtype=np.float32),
            object_state=np.zeros([
                track_length,
            ], dtype=int),
            lane=np.zeros([
                track_length,
            ], dtype=int),
        ),
        metadata=dict(
            track_length=track_length, type=MetaDriveType.TRAFFIC_LIGHT, object_id=object_id, dataset="waymo"
        )
    )

def save_as_metadrive_data(index, pred_i, other, save_path):
    from metadrive.scenario import ScenarioDescription as SD, MetaDriveType

    scenario = SD()

    scenario[SD.ID] = 'TrafficGen-{}'.format(index)
    track_len = len(pred_i)
    scenario[SD.LENGTH] = track_len
    scenario[SD.VERSION] = "2023-04-01"

    scenario['dynamic_map_states'] = [{}]

    scenario[SD.METADATA] = {}
    scenario[SD.METADATA][SD.TIMESTEP] = np.array([x / 10 for x in range(190)], dtype=np.float32)
    scenario[SD.METADATA][SD.SDC_ID] = str(0)
    scenario[SD.METADATA][SD.METADRIVE_PROCESSED] = False
    scenario[SD.METADATA][SD.COORDINATE] = MetaDriveType.COORDINATE_WAYMO

    # Tracks
    scenario[SD.TRACKS] = {}
    num_vehicles = pred_i.shape[1]  # pred_i in shape [T, #cars, 8]
    agent = pred_i
    for i in range(num_vehicles):

        agent_state = {}
        agent_i = agent[:, i]

        agent_state[SD.TYPE] = MetaDriveType.VEHICLE

        agent_state["metadata"] = {
            "object_id": str(i),
            "track_length": track_len,
            "type": MetaDriveType.VEHICLE,
        }
        agent_state["state"] = {}
        agent_state["state"]["position"] = agent_i[:, :2].astype(np.float32)
        agent_state["state"]["valid"] = np.ones([len(agent_i), ], dtype=np.bool)

        agent_state["state"]["size"] = np.ones([len(agent_i), 2], dtype=np.float32) * 2.332
        agent_state["state"]["size"][:, 0] *= 5.286  # length
        agent_state["state"]["size"][:, 1] *= 2.332  # width

        agent_state["state"]["heading"] = agent_i[:, 4].reshape(-1, 1).astype(np.float32)
        agent_state["state"]["velocity"] = agent_i[:, 2:4].astype(np.float32)

        scenario[SD.TRACKS][str(i)] = agent_state

        # state[:, :2] = agent_i[:, :2]
        # state[:, 3] = 5.286
        # state[:, 4] = 2.332
        # state[:, 7:9] = agent_i[:, 2:4]
        # state[:, -1] = 1
        # state[:, 6] = agent_i[:, 4]  # + np.pi / 2
        # track['state'] = state
        # scenario['tracks'][i] = track

    # center_info = other['center_info']

    # Map features
    scenario[SD.MAP_FEATURES] = {}
    lane = other['unsampled_lane']
    lane_id = np.unique(lane[..., -1]).astype(int)
    for id in lane_id:
        a_lane = {}
        id_set = lane[..., -1] == id
        points = lane[id_set]
        line_type = points[0, -2]
        a_lane['type'] = get_type_class(line_type)
        a_lane['polyline'] = points[:, :2]

        # The original data stored here has been discarded and can not be recovered.
        # This issue should be addressed in later implementation.
        a_lane["speed_limit_kmh"] = 1000

        scenario[SD.MAP_FEATURES][str(id)] = a_lane

    # Dynamics state
    scenario[SD.DYNAMIC_MAP_STATES] = {}
    traffic_light_compact_state = other["traf"]

    for step_count, step_state in enumerate(traffic_light_compact_state):
        if step_count >= track_len:
            break
        for traffic_light_state in step_state:
            lane_id = str(traffic_light_state[0])
            if lane_id not in scenario[SD.DYNAMIC_MAP_STATES]:
                scenario[SD.DYNAMIC_MAP_STATES][lane_id] = _traffic_light_state_template(lane_id, track_len)
            scenario[SD.DYNAMIC_MAP_STATES][lane_id]["state"]["lane"][step_count] = traffic_light_state[0]
            scenario[SD.DYNAMIC_MAP_STATES][lane_id]["state"]["stop_point"][step_count] = traffic_light_state[1:4]
            scenario[SD.DYNAMIC_MAP_STATES][lane_id]["state"]["object_state"][step_count] = traffic_light_state[4]

    scenario = scenario.to_dict()
    SD.sanity_check(scenario, check_self_type=True)

    with open(save_path, 'wb') as f:
        pickle.dump(scenario, f)

    print("MetaDrive-compatible scenario data is saved at: ", save_path)


class InitDataset(Dataset):
    """
    If in debug, it will load debug dataset
    """
    def __init__(self, cfg):
        self.total_data_usage = cfg["data_usage"]
        self.data_path = os.path.join(TRAFFICGEN_ROOT, cfg['data_path'])

        self.from_metadrive = cfg.get("from_metadrive", False)

        self.data_len = None
        self.data_loaded = {}
        self.cfg = cfg
        super(InitDataset, self).__init__()

    def load_data(self):

        data_path = self.data_path
        for i in range(self.total_data_usage):
            data_file_path = os.path.join(data_path, f'{i}.pkl')
            with open(data_file_path, 'rb+') as f:
                datas = pickle.load(f)

            if self.from_metadrive:
                from trafficgen.utils.get_md_data import metadrive_scenario_to_init_data
                datas = metadrive_scenario_to_init_data(datas)

            data = process_data_to_internal_format(datas)
            self.data_loaded[i] = data[0]

    def __len__(self):

        if not self.data_loaded:
            self.load_data()

        return self.total_data_usage

    def __getitem__(self, index):
        """
        Calculate for saving spaces
        """

        if not self.data_loaded:
            self.load_data()

        return self.data_loaded[index]

def get_vec_based_rep(case_info):

    thres = 5
    max_agent_num = 32
    # process future agent

    agent = case_info['agent']
    vectors = case_info["center"]

    agent_mask = case_info['agent_mask']

    vec_x = ((vectors[..., 0] + vectors[..., 2]) / 2)
    vec_y = ((vectors[..., 1] + vectors[..., 3]) / 2)

    agent_x = agent[..., 0]
    agent_y = agent[..., 1]

    b, vec_num = vec_y.shape
    _, agent_num = agent_x.shape

    vec_x = np.repeat(vec_x[:, np.newaxis], axis=1, repeats=agent_num)
    vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

    agent_x = np.repeat(agent_x[:, :, np.newaxis], axis=-1, repeats=vec_num)
    agent_y = np.repeat(agent_y[:, :, np.newaxis], axis=-1, repeats=vec_num)

    dist = np.sqrt((vec_x - agent_x)**2 + (vec_y - agent_y)**2)

    cent_mask = np.repeat(case_info['center_mask'][:, np.newaxis], axis=1, repeats=agent_num)
    dist[cent_mask == 0] = 10e5
    vec_index = np.argmin(dist, -1)
    min_dist_to_lane = np.min(dist, -1)
    min_dist_mask = min_dist_to_lane < thres

    selected_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], axis=1)

    vx, vy = agent[..., 2], agent[..., 3]
    v_value = np.sqrt(vx**2 + vy**2)
    low_vel = v_value < 0.1

    dir_v = np.arctan2(vy, vx)
    x1, y1, x2, y2 = selected_vec[..., 0], selected_vec[..., 1], selected_vec[..., 2], selected_vec[..., 3]
    dir = np.arctan2(y2 - y1, x2 - x1)
    agent_dir = agent[..., 4]

    v_relative_dir = cal_rel_dir(dir_v, agent_dir)
    relative_dir = cal_rel_dir(agent_dir, dir)

    v_relative_dir[low_vel] = 0

    v_dir_mask = abs(v_relative_dir) < np.pi / 6
    dir_mask = abs(relative_dir) < np.pi / 4

    agent_x = agent[..., 0]
    agent_y = agent[..., 1]
    vec_x = (x1 + x2) / 2
    vec_y = (y1 + y2) / 2

    cent_to_agent_x = agent_x - vec_x
    cent_to_agent_y = agent_y - vec_y

    coord = rotate(cent_to_agent_x, cent_to_agent_y, np.pi / 2 - dir)

    vec_len = np.clip(np.sqrt(np.square(y2 - y1) + np.square(x1 - x2)), a_min=4.5, a_max=5.5)

    lat_perc = np.clip(coord[..., 0], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len
    long_perc = np.clip(coord[..., 1], a_min=-vec_len / 2, a_max=vec_len / 2) / vec_len

    total_mask = min_dist_mask * agent_mask * v_dir_mask * dir_mask
    total_mask[:, 0] = 1
    total_mask = total_mask.astype(bool)

    b_s, agent_num, agent_dim = agent.shape
    agent_ = np.zeros([b_s, max_agent_num, agent_dim])
    agent_mask_ = np.zeros([b_s, max_agent_num]).astype(bool)

    the_vec = np.take_along_axis(vectors, vec_index[..., np.newaxis], 1)
    # 0: vec_index
    # 1-2 long and lat percent
    # 3-5 velocity and direction
    # 6-9 lane vector
    # 10-11 lane type and traff state
    info = np.concatenate(
        [
            vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis],
            v_value[..., np.newaxis], v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec
        ], -1
    )

    info_ = np.zeros([b_s, max_agent_num, info.shape[-1]])

    for i in range(agent.shape[0]):
        agent_i = agent[i][total_mask[i]]
        info_i = info[i][total_mask[i]]

        agent_i = agent_i[:max_agent_num]
        info_i = info_i[:max_agent_num]

        valid_num = agent_i.shape[0]
        agent_i = np.pad(agent_i, [[0, max_agent_num - agent_i.shape[0]], [0, 0]])
        info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])

        agent_[i] = agent_i
        info_[i] = info_i
        agent_mask_[i, :valid_num] = True

    # case_info['vec_index'] = info[...,0].astype(int)
    # case_info['relative_dir'] = info[..., 1]
    # case_info['long_perc'] = info[..., 2]
    # case_info['lat_perc'] = info[..., 3]
    # case_info['v_value'] = info[..., 4]
    # case_info['v_dir'] = info[..., 5]

    case_info['vec_based_rep'] = info_[..., 1:]
    case_info['agent_vec_index'] = info_[..., 0].astype(int)
    case_info['agent_mask'] = agent_mask_
    case_info["agent"] = agent_

    return

def transform_coordinate_map(data):
    """
    Every frame is different
    """
    timestep = data['all_agent'].shape[0]

    # sdc_theta = data['sdc_theta'][:,np.newaxis]
    ego = data['all_agent'][:, 0]
    pos = ego[:, [0, 1]][:, np.newaxis]

    lane = data['lane'][np.newaxis]
    lane = np.repeat(lane, timestep, axis=0)
    lane[..., :2] -= pos

    x = lane[..., 0]
    y = lane[..., 1]
    ego_heading = ego[:, [4]]
    lane[..., :2] = rotate(x, y, -ego_heading)

    unsampled_lane = data['unsampled_lane'][np.newaxis]
    unsampled_lane = np.repeat(unsampled_lane, timestep, axis=0)
    unsampled_lane[..., :2] -= pos

    x = unsampled_lane[..., 0]
    y = unsampled_lane[..., 1]
    ego_heading = ego[:, [4]]
    unsampled_lane[..., :2] = rotate(x, y, -ego_heading)
    return lane, unsampled_lane[0]

def process_agent(agent, sort_agent):

    ego = agent[:, 0]

    ego_pos = copy.deepcopy(ego[:, :2])[:, np.newaxis]
    ego_heading = ego[:, [4]]

    agent[..., :2] -= ego_pos
    agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
    agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
    agent[..., 4] -= ego_heading

    agent_mask = agent[..., -1]
    agent_type_mask = agent[..., -2] == 1
    agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)
    mask = agent_mask * agent_type_mask * agent_range_mask

    bs, agent_num, _ = agent.shape
    sorted_agent = np.zeros_like(agent)
    sorted_mask = np.zeros_like(agent_mask).astype(bool)
    sorted_agent[:, 0] = agent[:, 0]
    sorted_mask[:, 0] = True
    for i in range(bs):
        xy = copy.deepcopy(agent[i, 1:, :2])
        agent_i = copy.deepcopy(agent[i, 1:])
        mask_i = mask[i, 1:]

        # put invalid agent to the right down side
        xy[mask_i == False, 0] = 10e8
        xy[mask_i == False, 1] = -10e8

        raster = np.floor(xy / 0.25)
        raster = np.concatenate([raster, agent_i, mask_i[:, np.newaxis]], -1)
        y_index = np.argsort(-raster[:, 1])
        raster = raster[y_index]
        y_set = np.unique(raster[:, 1])[::-1]
        for y in y_set:
            ind = np.argwhere(raster[:, 1] == y)[:, 0]
            ys = raster[ind]
            x_index = np.argsort(ys[:, 0])
            raster[ind] = ys[x_index]
        # scene = np.delete(raster, [0, 1], axis=-1)
        sorted_agent[i, 1:] = raster[..., 2:-1]
        sorted_mask[i, 1:] = raster[..., -1]

    if sort_agent:
        return sorted_agent[..., :-1], sorted_mask
    else:
        agent_nums = np.sum(sorted_mask, axis=-1)
        for i in range(sorted_agent.shape[0]):
            agent_num = int(agent_nums[i])
            permut_idx = np.random.permutation(np.arange(1, agent_num)) - 1
            sorted_agent[i, 1:agent_num] = sorted_agent[i, 1:agent_num][permut_idx]
        return sorted_agent[..., :-1], sorted_mask

def get_gt(case_info):

    # 0: vec_index
    # 1-2 long and lat percent
    # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
    # 6-9 lane vector
    # 10-11 lane type and traff state
    center_num = case_info['center'].shape[1]
    lane_inp = case_info['lane_inp'][:, :center_num]
    agent_vec_index = case_info['agent_vec_index']
    vec_based_rep = case_info['vec_based_rep']
    bbox = case_info['agent'][..., 5:7]

    b, lane_num, _ = lane_inp.shape
    gt_distribution = np.zeros([b, lane_num])
    gt_vec_based_coord = np.zeros([b, lane_num, 5])
    gt_bbox = np.zeros([b, lane_num, 2])
    for i in range(b):
        mask = case_info['agent_mask'][i].sum()
        index = agent_vec_index[i].astype(int)
        gt_distribution[i][index[:mask]] = 1
        gt_vec_based_coord[i, index] = vec_based_rep[i, :, :5]
        gt_bbox[i, index] = bbox[i]
    case_info['gt_bbox'] = gt_bbox
    case_info['gt_distribution'] = gt_distribution
    case_info['gt_long_lat'] = gt_vec_based_coord[..., :2]
    case_info['gt_speed'] = gt_vec_based_coord[..., 2]
    case_info['gt_vel_heading'] = gt_vec_based_coord[..., 3]
    case_info['gt_heading'] = gt_vec_based_coord[..., 4]

def _process_map_inp(case_info):
    center = copy.deepcopy(case_info['center'])
    center[..., :4] /= RANGE
    edge = copy.deepcopy(case_info['bound'])
    edge[..., :4] /= RANGE
    cross = copy.deepcopy(case_info['cross'])
    cross[..., :4] /= RANGE
    rest = copy.deepcopy(case_info['rest'])
    rest[..., :4] /= RANGE

    case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
    case_info['lane_mask'] = np.concatenate(
        [case_info['center_mask'], case_info['bound_mask'], case_info['cross_mask'], case_info['rest_mask']],
        axis=1
    )
    return

def process_data_to_internal_format(data):
    case_info = {}
    gap = 20

    other = {}

    other['traf'] = data['traffic_light']

    agent = copy.deepcopy(data['all_agent'])
    data['all_agent'] = data['all_agent'][0:-1:gap]
    data['lane'], other['unsampled_lane'] = transform_coordinate_map(data)
    data['traffic_light'] = data['traffic_light'][0:-1:gap]

    other['lane'] = data['lane'][0]

    # transform agent coordinate
    ego = agent[:, 0]
    ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
    ego_heading = ego[[0], [4]]
    agent[..., :2] -= ego_pos
    agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
    agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
    agent[..., 4] -= ego_heading
    agent_mask = agent[..., -1]
    agent_type_mask = agent[..., -2]
    agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)
    mask = agent_mask * agent_type_mask * agent_range_mask

    agent = WaymoAgent(agent)
    other['gt_agent'] = agent.get_inp(act=True)
    other['gt_agent_mask'] = mask

    # process agent and lane data
    case_info["agent"], case_info["agent_mask"] = process_agent(data['all_agent'], False)
    case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
    case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
        data['lane'], data['traffic_light'], lane_range=RANGE, offest=0)

    # get vector-based representation
    get_vec_based_rep(case_info)

    agent = WaymoAgent(case_info['agent'], case_info['vec_based_rep'])

    case_info['agent_feat'] = agent.get_inp()

    _process_map_inp(case_info)

    get_gt(case_info)

    case_num = case_info['agent'].shape[0]
    case_list = []
    for i in range(case_num):
        dic = {}
        for k, v in case_info.items():
            dic[k] = v[i]
        case_list.append(dic)

    case_list[0]['other'] = other

    return case_list
