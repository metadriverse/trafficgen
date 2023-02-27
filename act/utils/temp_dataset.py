import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from utils.utils import wash
from utils.config import load_config_act, get_parsed_args
from utils.typedef import AgentType, RoadEdgeType, RoadLineType
from shapely.geometry import Polygon

LANE_SAMPLE = 10
RANGE = 60
MAX_AGENT = 32


class WaymoAgent:
    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:
            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])
            self.length_width = feature[..., 6:8]

        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    def get_agent(self, indx):
        return WaymoAgent(self.feature[[indx]], self.vec_based_info[[indx]])

    def get_inp(self, act=False, act_inp=False):

        if act:
            return np.concatenate([self.position, self.velocity, self.heading, self.length_width], axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate([pos, velo, cos_head, sin_head, self.length_width], axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        agent_feat = np.concatenate([pos, velo, cos_head, sin_head, self.length_width, vec_based_rep], axis=-1)
        return agent_feat

    def get_rect(self, pad=0):

        l, w = (self.length_width[..., 0] + pad) / 2, (self.length_width[..., 1] + pad) / 2
        x1, y1 = l, w
        x2, y2 = l, -w

        point1 = rotate(x1, y1, self.heading[..., 0])
        point2 = rotate(x2, y2, self.heading[..., 0])
        center = self.position

        x1, y1 = point1[..., [0]], point1[..., [1]]
        x2, y2 = point2[..., [0]], point2[..., [1]]

        p1 = np.concatenate([center[..., [0]] + x1, center[..., [1]] + y1], axis=-1)
        p2 = np.concatenate([center[..., [0]] + x2, center[..., [1]] + y2], axis=-1)
        p3 = np.concatenate([center[..., [0]] - x1, center[..., [1]] - y1], axis=-1)
        p4 = np.concatenate([center[..., [0]] - x2, center[..., [1]] - y2], axis=-1)

        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])

        agent_num, dim = p1.shape

        rect_list = []
        for i in range(agent_num):
            rect = np.stack([p1[i], p2[i], p3[i], p4[i]])
            rect_list.append(rect)
        return rect_list

    def get_polygon(self):
        rect_list = self.get_rect(pad=0.2)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a, b, c, d]))

        return poly_list


def cal_rel_dir(dir1, dir2):
    dist = dir1 - dir2

    while not np.all(dist >= 0):
        dist[dist < 0] += np.pi * 2
    while not np.all(dist < np.pi * 2):
        dist[dist >= np.pi * 2] -= np.pi * 2

    dist[dist > np.pi] -= np.pi * 2
    return dist


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def process_lane(lane, max_vec, lane_range, offset=-40):
    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0] + offset) < lane_range) * (abs(lane[..., 1]) < lane_range)

    lane_id = np.unique(lane[..., -2]).astype(int)

    vec_list = []
    vec_mask_list = []
    b_s, _, lane_dim = lane.shape

    for id in lane_id:
        id_set = lane[..., -2] == id
        points = lane[id_set].reshape(b_s, -1, lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s, -1)

        vector = np.zeros([b_s, points.shape[1] - 1, vec_dim])
        vector[..., 0:2] = points[:, :-1, :2]
        vector[..., 2:4] = points[:, 1:, :2]
        # id
        # vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:, 1:, 2]
        # traffic light
        vector[..., 5] = points[:, 1:, 4]
        vec_mask = masks[:, :-1] * masks[:, 1:]
        vector[vec_mask == 0] = 0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list, axis=1) if vec_list else np.zeros([b_s, 0, vec_dim])
    vector_mask = np.concatenate(vec_mask_list, axis=1) if vec_mask_list else np.zeros([b_s, 0], dtype=bool)

    all_vec = np.zeros([b_s, max_vec, vec_dim])
    all_mask = np.zeros([b_s, max_vec])
    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]

        dist = vector_t[..., 0]**2 + vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        all_vec[t] = vector_t
        all_mask[t] = mask_t

    return all_vec, all_mask.astype(bool)


def process_map(lane, traf, center_num=384, edge_num=128, lane_range=60, offest=-40):
    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_id = lane[..., -1]
    b_s = lane_id.shape[0]

    for i in range(b_s):
        traf_t = traf[i]
        lane_id_t = lane_id[i]
        for a_traf in traf_t:
            control_lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_id_t == control_lane_id)
            lane_with_traf[i, lane_idx, -1] = state

    # lane = np.delete(lane_with_traf,-2,axis=-1)
    lane = lane_with_traf
    lane_type = lane[0, :, 2]
    center_1 = lane_type == 1
    center_2 = lane_type == 2
    center_3 = lane_type == 3
    center_ind = center_1 + center_2 + center_3

    boundary_1 = lane_type == 15
    boundary_2 = lane_type == 16
    bound_ind = boundary_1 + boundary_2

    cross_walk = lane_type == 18
    speed_bump = lane_type == 19
    cross_ind = cross_walk + speed_bump

    rest = ~(center_ind + bound_ind + cross_walk + speed_bump + cross_ind)

    cent, cent_mask = process_lane(lane[:, center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[:, bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[:, cross_ind], 32, lane_range, offest)
    rest, rest_mask = process_lane(lane[:, rest], 192, lane_range, offest)

    return cent, cent_mask, bound, bound_mask, cross, cross_mask, rest, rest_mask


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


class actDataset(Dataset):
    """
    If in debug, it will load debug dataset
    """
    def __init__(self, cfg, args=None, eval=False):
        self.total_data_usage = cfg["data_usage"]
        self.data_path = cfg['data_path']
        self.pred_len = cfg['pred_len']

        self.eval = eval
        self.data_len = None
        self.data_loaded = {}
        self.scene_data = {}
        self.cfg = cfg
        self.load_data()
        super(actDataset, self).__init__()

    def load_data(self):
        for i in tqdm(range(self.total_data_usage)):
            data_file_path = os.path.join(self.data_path, f'{i}.pkl')

            with open(data_file_path, 'rb+') as f:
                datas = pickle.load(f)
            datas = self.process(datas)
            wash(datas)
            self.data_loaded[i] = datas

        data_path = os.path.join(self.data_path, 'act_cache.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(self.data_loaded, f)

        self.data_len = self.total_data_usage

    def __len__(self):
        # debug set length=478
        return self.data_len

    def __getitem__(self, index):
        """
        Calculate for saving spaces
        """
        return self.data_loaded[index]

    def process_scene(self, data):
        case = {}

        sdc_theta = data['sdc_theta']
        pos = data['sdc_pos']
        all_agent = np.concatenate([data['ego_p_c_f'][np.newaxis], data['nbrs_p_c_f']], axis=0)
        coord = self.rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
        vel = self.rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)
        yaw = -sdc_theta + np.pi / 2
        all_agent[..., 4] = all_agent[..., 4] + yaw
        all_agent[..., :2] = coord
        all_agent[..., 2:4] = vel
        pred_list = np.append(np.array([1]), data['pred_list']).astype(bool)
        all_agent = all_agent[pred_list][:, 0]

        valid_mask = all_agent[..., -1] == 1.
        type_mask = all_agent[:, -2] == 1.
        mask = valid_mask * type_mask
        all_agent = all_agent[mask]

        case['all_agent'] = all_agent
        case['lane'] = data['lane']
        case['traf'] = data['traf_p_c_f']
        return case

    def rotate(self, x, y, angle):
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        return output_coords

    def process_agent(self, data):
        agent = data['all_agent']
        ego = agent[:, 0]

        ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        ego_heading = ego[[0], [4]]

        agent[..., :2] -= ego_pos
        agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        agent[..., 4] -= ego_heading

        agent_mask = agent[..., -1]
        agent_type_mask = agent[..., -2]
        agent_range_mask = (abs(agent[..., 0] - 40) < RANGE) * (abs(agent[..., 1]) < RANGE)
        mask = agent_mask * agent_type_mask * agent_range_mask

        return agent, mask.astype(bool)

    def get_inp_gt(self, case_info, agent, agent_mask):
        agent_context = agent[0]
        agent_mask = agent_mask[0]
        agent_context = agent_context[agent_mask]
        agent_context = agent_context[:MAX_AGENT]
        agent_mask = agent_mask[:MAX_AGENT]
        agent_context = WaymoAgent(agent_context)
        agent_context = agent_context.get_inp(act_inp=True)
        agent_context = np.pad(agent_context, ([0, MAX_AGENT - agent_context.shape[0]], [0, 0]))
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_mask.shape[0]]))

        case_info['agent'] = agent_context
        case_info['agent_mask'] = agent_mask

        ego_future = agent[:self.pred_len, 0]

        case_info['gt_pos'] = ego_future[1:, :2]  # -ego_future[:-1,:2]
        case_info['gt_vel'] = ego_future[1:, 2:4]
        case_info['gt_heading'] = cal_rel_dir(ego_future[1:, 4], 0)

        return agent_context, agent_mask, agent[:self.pred_len, 0]

    def transform_coordinate_map(self, data):
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

        data['lane'] = lane

    def process(self, data):
        case_info = {}

        self.transform_coordinate_map(data)
        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'][[0]], [data['traffic_light'][0]], center_num=256, edge_num=128, offest=-40, lane_range=60)

        case_info['center'] = case_info['center'][0]
        case_info['center_mask'] = case_info['center_mask'][0]
        case_info['bound'] = case_info['bound'][0]
        case_info['bound_mask'] = case_info['bound_mask'][0]
        case_info['cross'] = case_info['cross'][0]
        case_info['cross_mask'] = case_info['cross_mask'][0]
        case_info['rest'] = case_info['rest'][0]
        case_info['rest_mask'] = case_info['rest_mask'][0]

        agent, agent_mask = self.process_agent(data)
        self.get_inp_gt(case_info, agent, agent_mask)

        return case_info


if __name__ == "__main__":
    print('loading data...')
    args = get_parsed_args()
    cfg = load_config_act(args.config)
    cfg['use_cache'] = False
    actDataset(cfg)
