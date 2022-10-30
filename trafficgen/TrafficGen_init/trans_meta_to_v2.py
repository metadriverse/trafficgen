import os
import pickle
from enum import Enum

import numpy as np


def yaw_to_y(angle):
    angle = trans_angle(angle)
    angle_to_y = angle - np.pi / 2
    angle_to_y = -1 * angle_to_y

    return angle_to_y


def yaw_to_theta(angle, theta):
    """
    In time horizon
    """

    theta = trans_angle(theta)
    angle -= theta
    for i in range(len(angle)):
        angle[i] = trans_angle(angle[i])
    angle[angle >= np.pi] -= 2 * np.pi

    return angle


def trans_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def extract_boundaries(fb):
    b = []
    # b = np.zeros([len(fb), 4], dtype='int64')
    for k in range(len(fb)):
        c = dict()
        c['index'] = [fb[k].lane_start_index, fb[k].lane_end_index]
        c['type'] = RoadLineType(fb[k].boundary_type)
        c['id'] = fb[k].boundary_feature_id
        b.append(c)

    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nb = dict()
        nb['id'] = fb[k].feature_id
        nb['indexes'] = [
            fb[k].self_start_index, fb[k].self_end_index, fb[k].neighbor_start_index, fb[k].neighbor_end_index
        ]
        nb['boundaries'] = extract_boundaries(fb[k].boundaries)
        nb['id'] = fb[k].feature_id
        nbs.append(nb)
    return nbs


def transform_coord(coords, angle):
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)

    if coords.shape[1] == 3:
        output_coords = np.concatenate((output_coords, coords[:, 2:]), axis=-1)
    return output_coords


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

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineType else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineType.SOLID_DOUBLE_YELLOW, RoadLineType.PASSING_DOUBLE_YELLOW, RoadLineType.SOLID_SINGLE_YELLOW,
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineType.BROKEN_DOUBLE_YELLOW, RoadLineType.BROKEN_SINGLE_YELLOW, RoadLineType.BROKEN_SINGLE_WHITE
        ] else False


class RoadEdgeType(Enum):
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeType else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeType.BOUNDARY else False


class AgentType(Enum):
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


if __name__ == '__main__':
    data_path = '/Users/fenglan/Downloads/waymo/100_training'

    with open('/Users/fenglan/Downloads/waymo/v2_data/0.pkl', 'rb+') as f:
        datas = pickle.load(f)

    for i in range(100):
        path = os.path.join(data_path, f'{i}.pkl')
        with open(path, 'rb+') as f:
            data = pickle.load(f)

        output = dict()
        output['id'] = data['id']

        agents = []
        sdc_id = data['sdc_index']
        for k, agent in data['tracks'].items():
            agent_type = agent['type'].value
            if not agent_type == 1:
                continue
            state = agent['state']
            # x,y,vx,vy,yaw,l,w,type,validity

            index = [0, 1, 7, 8, 6, 3, 4]
            selected = state[:, index]
            agent_type = np.ones(selected.shape[0])
            selected = np.concatenate([selected, agent_type[:, np.newaxis], state[:, -1][:, np.newaxis]], -1)
            if k == sdc_id:
                agents.insert(0, selected)
            else:
                agents.append(selected)

        agent_all = np.stack(agents)[:, :190]
        egos = []
        nbrs = []
        map = []
        map_mask = []
        sdcs = []

        for i in range(190):
            agents = agent_all[:, i]
            sdc_x, sdc_y, sdc_yaw = agents[0, 0], agents[0, 1], agents[0, 4]
            sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
            agents[:, 0] -= sdc_x
            agents[:, 1] -= sdc_y
            agents[:, :2] = transform_coord(agents[:, :2], sdc_theta)
            agents[:, 2:4] = transform_coord(agents[:, 2:4], sdc_theta)
            agents[:, 4] = yaw_to_theta(agents[:, 4], sdc_yaw)

            egos.append(agents[0])
            nbrs.append(agents[1:])

            lanes = []
            for k, lane in input['map'].items():
                a_lane = np.zeros([20, 4])
                tp = 0
                try:
                    lane_type = lane['type']
                except:
                    lane_type = lane['sign']
                    poly_line = lane['polygon']
                    if lane_type == 'cross_walk':
                        tp = 18
                    elif lane_type == 'speed_bump':
                        tp = 19

                if lane_type == 'center_lane':
                    poly_line = lane['polyline']
                    tp = 1

                elif lane_type == RoadEdgeType.BOUNDARY or lane_type == RoadEdgeType.MEDIAN:
                    tp = 15 if lane_type == RoadEdgeType.BOUNDARY else 16
                    poly_line = lane['polyline']
                elif 'polyline' in lane:
                    tp = 7
                    poly_line = lane['polyline']
                if tp == 0:
                    continue

                a_lane[:, 2] = tp

                a_lane[:, :2] = poly_line
                lanes.append(a_lane)

            lanes = np.stack(lanes)

            pos = np.stack([sdc_x, sdc_y], axis=-1)
            lanes[..., :2] -= pos
            lanes[..., :2] = transform_coord(lanes[..., :2], sdc_theta)

            # lanes[abs(lanes[..., 1]) > 80, -1] = 0
            # lanes[abs(lanes[..., 2]) > 80, -1] = 0
            valid_ret = np.sum(lanes[..., -1], -1)
            lane_mask = valid_ret.astype(bool)
            map.append(lanes)
            map_mask.append(lane_mask)
            sdcs.append([sdc_x, sdc_y, sdc_yaw, sdc_theta])

        output['ego_p_c_f'] = np.stack(egos)
        output['nbrs_p_c_f'] = np.stack(nbrs).transpose(1, 0, 2)
        output['lane'] = np.stack(map)
        output['lane_mask'] = np.stack(map_mask)
