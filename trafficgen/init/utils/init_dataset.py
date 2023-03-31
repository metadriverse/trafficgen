import copy
import os
import pickle

import numpy as np
import torch
from shapely.geometry import Polygon
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from trafficgen.utils.config import load_config_init, get_parsed_args
from trafficgen.utils.utils import cal_rel_dir, rotate, process_map, wash


def get_agent_pos_from_vec(vec, long_lat, speed, vel_heading, heading, bbox):
    x1, y1, x2, y2 = vec[:, 0], vec[:, 1], vec[:, 2], vec[:, 3]
    x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2

    vec_len = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len * long_lat[..., 0]
    lat_pos = vec_len * long_lat[..., 1]

    coord = rotate(lat_pos, long_pos, -np.pi / 2 + vec_dir)

    coord[:, 0] += x_center
    coord[:, 1] += y_center

    agent_dir = vec_dir + heading
    v_dir = vel_heading + agent_dir

    vel = torch.stack([torch.cos(v_dir) * speed, torch.sin(v_dir) * speed], axis=-1)
    agent_num, _ = vel.shape

    type = Tensor([[1]]).repeat(agent_num, 1).to(coord.device)
    agent = torch.cat([coord, vel, agent_dir.unsqueeze(1), bbox, type], dim=-1).detach().cpu().numpy()

    vec_based_rep = torch.cat(
        [long_lat, speed.unsqueeze(-1),
         vel_heading.unsqueeze(-1),
         heading.unsqueeze(-1), vec], dim=-1
    ).detach().cpu().numpy()

    agent = WaymoAgent(agent, vec_based_rep)

    return agent


def get_gt(case_info):
    # 0: vec_index
    # 1-2 long and lat percent
    # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
    # 6-9 lane vector
    # 10-11 lane type and traff state
    center_num = case_info['center'].shape[1]

    lane_inp = case_info['lane_inp'][:, :center_num]
    agent_vec_indx = case_info['agent_vec_indx']
    vec_based_rep = case_info['vec_based_rep']
    bbox = case_info['agent'][..., 5:7]

    b, lane_num, _ = lane_inp.shape
    gt_distribution = np.zeros([b, lane_num])
    gt_vec_based_coord = np.zeros([b, lane_num, 5])
    gt_bbox = np.zeros([b, lane_num, 2])
    for i in range(b):
        mask = case_info['agent_mask'][i].sum()
        indx = agent_vec_indx[i].astype(int)
        gt_distribution[i][indx[:mask]] = 1
        gt_vec_based_coord[i, indx] = vec_based_rep[i, :, :5]
        gt_bbox[i, indx] = bbox[i]
    case_info['gt_bbox'] = gt_bbox
    case_info['gt_distribution'] = gt_distribution
    case_info['gt_long_lat'] = gt_vec_based_coord[..., :2]
    case_info['gt_speed'] = gt_vec_based_coord[..., 2]
    case_info['gt_vel_heading'] = gt_vec_based_coord[..., 3]
    case_info['gt_heading'] = gt_vec_based_coord[..., 4]


def transform_coordinate_map(data):
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
    return lane


def get_vec_rep(case_info):
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
            vec_index[..., np.newaxis], long_perc[..., np.newaxis], lat_perc[..., np.newaxis], v_value[..., np.newaxis],
            v_relative_dir[..., np.newaxis], relative_dir[..., np.newaxis], the_vec
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
    case_info['agent_vec_indx'] = info_[..., 0].astype(int)
    case_info['agent_mask'] = agent_mask_
    case_info["agent"] = agent_

    return


def process_agent(agent, RANGE=50, sort_agent=True):
    ego = agent[:, 0]

    ego_pos = copy.deepcopy(ego[:, :2])[:, np.newaxis]
    ego_heading = ego[:, [4]]

    agent[..., :2] -= ego_pos
    agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
    agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
    agent[..., 4] -= ego_heading

    agent_mask = agent[..., -1]
    agent_type_mask = (agent[..., -2] == 1)
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


def process_map_inp(case_info, map_size=50):
    center = copy.deepcopy(case_info['center'])
    center[..., :4] /= map_size
    edge = copy.deepcopy(case_info['bound'])
    edge[..., :4] /= map_size
    cross = copy.deepcopy(case_info['cross'])
    cross[..., :4] /= map_size
    rest = copy.deepcopy(case_info['rest'])
    rest[..., :4] /= map_size

    case_info['lane_inp'] = np.concatenate([center, edge, cross, rest], axis=1)
    case_info['lane_mask'] = np.concatenate(
        [case_info['center_mask'], case_info['bound_mask'], case_info['cross_mask'], case_info['rest_mask']], axis=1
    )
    return


class WaymoAgent:
    def __init__(self, feature, vec_based_info=None, range=50, max_speed=30, from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:

            self.position = feature[..., :2] * self.RANGE
            self.velocity = feature[..., 2:4] * self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5], feature[..., 4])[..., np.newaxis]
            self.length_width = feature[..., 6:8]
            type = np.ones_like(self.heading)
            self.feature = np.concatenate(
                [self.position, self.velocity, self.heading, self.length_width, type], axis=-1
            )
            if vec_based_info is not None:
                vec_based_rep = copy.deepcopy(vec_based_info)
                vec_based_rep[..., 5:9] *= self.RANGE
                vec_based_rep[..., 2] *= self.MAX_SPEED
                self.vec_based_info = vec_based_rep

        else:
            self.feature = feature
            self.position = feature[..., :2]
            self.velocity = feature[..., 2:4]
            self.heading = feature[..., [4]]
            self.length_width = feature[..., 5:7]
            self.type = feature[..., [7]]
            self.vec_based_info = vec_based_info

    @staticmethod
    def from_list_to_array(inp_list):
        MAX_AGENT = 32
        agent = np.concatenate([x.get_inp(act=True) for x in inp_list], axis=0)
        agent = agent[:MAX_AGENT]
        agent_num = agent.shape[0]
        agent = np.pad(agent, ([0, MAX_AGENT - agent_num], [0, 0]))
        agent_mask = np.zeros([agent_num])
        agent_mask = np.pad(agent_mask, ([0, MAX_AGENT - agent_num]))
        agent_mask[:agent_num] = 1
        agent_mask = agent_mask.astype(bool)
        return agent, agent_mask

    def get_agent(self, index):
        return WaymoAgent(self.feature[[index]], self.vec_based_info[[index]])

    def get_list(self):
        bs, agent_num, feature_dim = self.feature.shape
        vec_dim = self.vec_based_info.shape[-1]
        feature = self.feature.reshape([-1, feature_dim])
        vec_rep = self.vec_based_info.reshape([-1, vec_dim])
        agent_num = feature.shape[0]
        lis = []
        for i in range(agent_num):
            lis.append(WaymoAgent(feature[[i]], vec_rep[[i]]))
        return lis

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
        rect_list = self.get_rect(pad=0.25)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a, b, c, d]))

        return poly_list


LANE_SAMPLE = 10


class initDataset(Dataset):
    """
    If in debug, it will load debug dataset
    """
    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        self.cfg = cfg
        self.data_loaded = {}
        self.load_data()
        super(initDataset, self).__init__()

    def load_data(self):
        if self.cfg['use_cache']:
            data_path = os.path.join(self.data_path, 'init_cache.pkl')
            with open(data_path, 'rb+') as f:
                self.data_loaded = pickle.load(f)
            self.data_len = len(self.data_loaded)

        else:
            cnt = 0
            for file_indx in tqdm(range(self.cfg['data_usage'])):
                data_path = self.data_path
                data_file_path = os.path.join(data_path, f'{file_indx}.pkl')
                # file_indx+=1
                with open(data_file_path, 'rb+') as f:
                    datas = pickle.load(f)
                data = self.process(datas)
                case_cnt = 0
                for i in range(len(data)):
                    wash(data[i])
                    agent_num = data[i]['agent_mask'].sum()
                    if agent_num < self.cfg['min_agent']:
                        continue
                    self.data_loaded[cnt + case_cnt] = data[i]
                    case_cnt += 1
                cnt += case_cnt
            self.data_len = cnt

            # save cache
            data_path = os.path.join(self.data_path, 'init_cache.pkl')
            with open(data_path, 'wb') as f:
                pickle.dump(self.data_loaded, f)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        Calculate for saving spaces
        """
        return self.data_loaded[index]

    def process(self, data):
        case_info = {}

        map_size = self.cfg['map_size']
        gap = self.cfg['sample_gap']

        # sample original data in a fixed interval
        data['all_agent'] = data['all_agent'][0:-1:gap]
        data['traffic_light'] = data['traffic_light'][0:-1:gap]

        data['lane'] = transform_coordinate_map(data)

        case_info["agent"], case_info["agent_mask"] = process_agent(data['all_agent'], map_size, False)

        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'], case_info['rest_mask'] = process_map(
            data['lane'], data['traffic_light'], lane_range=self.cfg['map_size'], offest=0)

        get_vec_rep(case_info)

        agent = WaymoAgent(case_info['agent'], case_info['vec_based_rep'])

        case_info['agent_feat'] = agent.get_inp()

        process_map_inp(case_info, map_size)

        get_gt(case_info)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k, v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)

        return case_list


if __name__ == "__main__":
    args = get_parsed_args()
    cfg = load_config_init(args.config)
    cfg['use_cache'] = False
    initDataset(cfg)
