import datetime
import time
import numpy as np
import copy
from torch import Tensor
from utils.typedef import RoadLineType,RoadEdgeType

def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret = fn(*args, **kwargs)
        return ret, time.clock() - start

    return _wrapper


def get_time_str():
    return datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

def rotate(x, y, angle):
    other_x_trans = np.cos(angle) * x - np.sin(angle) * y
    other_y_trans = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords

def from_list_to_batch(inp_list):
    keys = inp_list[0].keys()

    batch = {}
    for key in keys:
        one_item = [item[key] for item in inp_list]
        batch[key] = Tensor(np.stack(one_item))

    return batch

def transform_to_agent(agent_i,agent,lane):

    all_ = copy.deepcopy(agent)

    center = copy.deepcopy(agent_i[:2])
    center_yaw = copy.deepcopy(agent_i[4])
    rotate_theta = -(center_yaw - np.pi / 2)

    all_[..., :2] -= center

    coord = rotate(all_[..., 0], all_[..., 1], rotate_theta)
    vel = rotate(all_[..., 2], all_[..., 3], rotate_theta)
    all_[..., :2] = coord
    all_[..., 2:4] = vel
    all_[..., 4] = all_[..., 4] - center_yaw
    # then recover lane's position
    lane = copy.deepcopy(lane)
    lane[..., :2] -= center
    output_coords = rotate(lane[..., 0], lane[..., 1], rotate_theta)
    lane[..., :2] = output_coords

    return all_, lane

def get_type_class(line_type):
    if line_type in range(1,4):
        return 'center_lane'
    elif line_type == 6:
        return RoadLineType.BROKEN_SINGLE_WHITE
    elif line_type == 7:
        return RoadLineType.SOLID_SINGLE_WHITE
    elif line_type == 8:
        return RoadLineType.SOLID_DOUBLE_WHITE
    elif line_type == 9:
        return RoadLineType.BROKEN_SINGLE_YELLOW
    elif line_type == 10:
        return RoadLineType.BROKEN_DOUBLE_YELLOW
    elif line_type == 11:
        return RoadLineType.SOLID_SINGLE_YELLOW
    elif line_type == 12:
        return RoadLineType.SOLID_DOUBLE_YELLOW
    elif line_type == 13:
        return RoadLineType.PASSING_DOUBLE_YELLOW
    elif line_type == 15:
        return RoadEdgeType.BOUNDARY
    elif line_type == 16:
        return RoadEdgeType.MEDIAN
    else:
        return 'other'