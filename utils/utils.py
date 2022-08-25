import datetime
import time
import numpy as np
import copy
from torch import Tensor
import torch
from utils.typedef import RoadLineType,RoadEdgeType
from shapely.geometry import Polygon

def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret = fn(*args, **kwargs)
        return ret, time.clock() - start

    return _wrapper

def get_polygon(center, yaw, L, W):

    l, w = L / 2, W / 2
    yaw+=torch.pi/2
    theta = torch.atan(w / l)
    s1 = torch.sqrt(l ** 2 + w ** 2)
    x1 = abs(torch.cos(theta + yaw) * s1)
    y1 = abs(torch.sin(theta + yaw) * s1)
    x2 = abs(torch.cos(theta - yaw) * s1)
    y2 = abs(torch.sin(theta - yaw) * s1)

    p1 = [center[0] + x1, center[1] +y1]
    p2 = [center[0] + x2, center[1] - y2]
    p3 = [center[0] - x1, center[1] - y1]
    p4 = [center[0] - x2, center[1] + y2]
    return Polygon([p1, p3, p2, p4])

def get_agent_pos_from_vec(vec,pred):
    long_perc,lat_perc,speed,v_dir, dir = pred[...,0],pred[...,1],pred[...,2],pred[...,3],pred[...,4]

    x1,y1,x2,y2 = vec[:,0],vec[:,1],vec[:,2],vec[:,3]
    vec_len = ((x1-x2)**2+(y1-y2)**2)**0.5

    vec_dir = torch.atan2(y2 - y1, x2 - x1)

    long_pos = vec_len*long_perc
    lat_pos = lat_perc*5

    coord = Tensor(rotate(lat_pos,long_pos,-vec_dir))

    coord[:,0]+=x1
    coord[:,1]+=y1

    agent_dir = vec_dir+dir
    v_dir=v_dir+agent_dir
    #v_dir = agent_dir

    vel = torch.stack([torch.cos(v_dir)*speed,torch.sin(v_dir)*speed], axis=-1)
    #dir_ = np.stack([np.cos(agent_dir),np.sin(agent_dir)], axis=-1)
    return coord, agent_dir,vel

def process_lane(lane,  max_vec,lane_range,offset = -40):

    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    vec_dim = 6

    lane_point_mask = (abs(lane[..., 0]) < lane_range) * (abs(lane[..., 1] + offset) < lane_range)

    lane_id = np.unique(lane[...,-2]).astype(int)

    vec_list = []
    vec_mask_list = []
    b_s, _, lane_dim= lane.shape

    for id in lane_id:
        id_set = lane[...,-2]==id
        points = lane[id_set].reshape(b_s,-1,lane_dim)
        masks = lane_point_mask[id_set].reshape(b_s,-1)

        vector = np.zeros([b_s,points.shape[1]-1,vec_dim])
        vector[..., 0:2] = points[:,:-1, :2]
        vector[..., 2:4] = points[:,1:, :2]
        # id
        #vector[..., 4] = points[:,1:, 3]
        # type
        vector[..., 4] = points[:,1:, 2]
        # traffic light
        vector[..., 5] = points[:,1:, 4]
        vec_mask = masks[:,:-1]*masks[:,1:]
        vector[vec_mask==0]=0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list,axis=1) if vec_list else np.zeros([0,0,vec_dim])
    vector_mask = np.concatenate(vec_mask_list,axis=1) if vec_mask_list else np.zeros([0,0],dtype=bool)

    all_vec = np.zeros([b_s,max_vec,vec_dim])
    all_mask = np.zeros([b_s,max_vec])
    for t in range(b_s):
        mask_t = vector_mask[t]
        vector_t = vector[t][mask_t]

        dist = vector_t[..., 0]**2+vector_t[..., 1]**2
        idx = np.argsort(dist)
        vector_t = vector_t[idx]
        mask_t = np.ones(vector_t.shape[0])

        vector_t = vector_t[:max_vec]
        mask_t = mask_t[:max_vec]

        vector_t = np.pad(vector_t, ([0, max_vec - vector_t.shape[0]], [0, 0]))
        mask_t = np.pad(mask_t, ([0, max_vec - mask_t.shape[0]]))
        all_vec[t] = vector_t
        all_mask[t] = mask_t

    return all_vec,all_mask.astype(bool)

def process_map(lane,traf, center_num=128, edge_num=128, lane_range=60, offest=-40):

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
            lane_with_traf[i,lane_idx, -1] = state

    #lane = np.delete(lane_with_traf,-2,axis=-1)
    lane = lane_with_traf
    lane_type = lane[0,:, 2]
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

    cent, cent_mask = process_lane(lane[:,center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[:,bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[:,cross_ind], 64, lane_range, offest)
    rest, rest_mask = process_lane(lane[:,rest], center_num, lane_range, offest)

    return cent, cent_mask, bound, bound_mask, cross, cross_mask, rest,rest_mask

def get_time_str():
    return datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

def cal_rel_dir(dir1,dir2):
    dist = dir1-dir2

    while not np.all(dist>=0):
        dist[dist<0]+=np.pi*2
    while not np.all(dist<np.pi*2):
        dist[dist>=np.pi*2]-=np.pi*2

    dist[dist>np.pi] -= np.pi*2
    return dist

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
    if isinstance(lane,Tensor):
        output_coords = Tensor(output_coords)
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