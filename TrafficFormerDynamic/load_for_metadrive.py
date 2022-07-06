import copy

import numpy as np
from enum import Enum
import torch
from l5kit.configs import load_config_data
from tqdm import tqdm
import pickle
from drivingforce.TrafficTranformer.utils.train import Trainer
from drivingforce.TrafficTranformer.utils.arg_parse import get_parsed_args
import os
from drivingforce.TrafficTranformer.load_and_draw import draw

TIME_START = 90

def down_sampling(line, sample_point_num,type=0):
    points_vector = np.zeros([sample_point_num, 2], dtype=np.float32)

    point_num = len(line)
    if point_num < sample_point_num:
        points_vector[:point_num] = line[:, :2]
        # closed polygon for crosswalk
        if type == 18 or 19:
            points_vector[point_num] = points_vector[0]
            point_num+=1
            points_vector[point_num:] = np.tile(line[0, :2], (sample_point_num - point_num, 1))
        else:
            points_vector[point_num:] = np.tile(line[point_num,:2], (sample_point_num - point_num, 1))
    else:
        interval = 1.0 * (point_num - 1) / sample_point_num
        selected_point_index = [int(np.round(i * interval)) for i in range(1, sample_point_num - 1)]
        selected_point_index = [0] + selected_point_index + [point_num - 1]
        selected_point = line[selected_point_index, :]
        points_vector[:, :2] = selected_point[:, :]
    return points_vector

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
    angle[angle>=np.pi]-=2 * np.pi

    return angle

    # transform angles into 0~2pi


def trans_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


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

def from_meta_to_tt(input):
    output = dict()
    output['id'] = input['id']

    agents = []
    sdc_id = input['sdc_index']
    for k,agent in input['tracks'].items():
            agent_type = agent['type'].value
            if not agent_type == 1:
                continue
            state = agent['state']
            # x,y,vx,vy,yaw,l,w,type,validity

            index = [0,1,7,8,6,3,4]
            selected = state[:,index]
            agent_type = np.ones(selected.shape[0])
            selected = np.concatenate([selected,agent_type[:,np.newaxis],state[:,-1][:,np.newaxis]],-1)
            if k == sdc_id:
                agents.insert(0,selected)
            else:
                agents.append(selected)

    agent_all = np.stack(agents)
    egos = []
    nbrs = []
    map = []
    map_mask = []
    sdcs = []

    for i in range(190):
        agents = agent_all[:,i]
        sdc_x, sdc_y, sdc_yaw = agents[0,0],agents[0,1],agents[0,4]
        sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
        agents[:,0]-=sdc_x
        agents[:,1]-=sdc_y
        agents[:,:2] = transform_coord(agents[:,:2], sdc_theta)
        agents[:, 2:4] = transform_coord(agents[:, 2:4], sdc_theta)
        agents[:,4] = yaw_to_theta(agents[:,4], sdc_yaw)

        egos.append(agents[0])
        nbrs.append(agents[1:])

        lanes = []
        for k, lane in input['map'].items():
            a_lane = np.zeros([20,4])
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
                tp = 15 if lane_type==RoadEdgeType.BOUNDARY else 16
                poly_line = lane['polyline']
            elif 'polyline' in lane:
                tp = 7
                poly_line = lane['polyline']
            if tp ==0:
                continue

            a_lane[:,2] = tp

            poly_line = down_sampling(poly_line[:,:2],20,tp)
            a_lane[:,:2] = poly_line
            lanes.append(a_lane)

        lanes = np.stack(lanes)

        pos = np.stack([sdc_x, sdc_y], axis=-1)
        lanes[...,:2]-=pos
        lanes[..., :2] = transform_coord(lanes[..., :2],sdc_theta)

        # lanes[abs(lanes[..., 1]) > 80, -1] = 0
        # lanes[abs(lanes[..., 2]) > 80, -1] = 0
        valid_ret = np.sum(lanes[..., -1], -1)
        lane_mask = valid_ret.astype(bool)
        map.append(lanes)
        map_mask.append(lane_mask)
        sdcs.append([sdc_x,sdc_y,sdc_yaw,sdc_theta])

    output['ego_p_c_f'] = np.stack(egos)
    output['nbrs_p_c_f'] = np.stack(nbrs).transpose(1,0,2)
    output['lane'] = np.stack(map)
    output['lane_mask'] = np.stack(map_mask)
    return output,sdcs

def load_for_metadrive(model_path,validation_path,cfg_path,tt_path,pred_data_path,data_num,data_aug):

    sdc_ = []
    datas = []
    for i in tqdm(range(data_num)):
        path = os.path.join(validation_path,f'{i}.pkl')
        with open(path, 'rb+') as f:
            data = pickle.load(f)
            processed,sdc_info = from_meta_to_tt(data)

            path = os.path.join(tt_path,f'{i}.pkl')
            with open(path,"wb") as f:
                pickle.dump(processed,f)

            sdc_.append(sdc_info)
            datas.append(data)

    args = get_parsed_args()
    args.distributed = False
    cfg = load_config_data(cfg_path)
    cfg['data_usage'] = data_num
    cfg['eval_data_usage'] = data_num
    cfg['eval_data_start_index'] = 0
    cfg['eval_batch_size'] = 16
    cfg['data_path'] = tt_path
    trainer = Trainer(args=args, cfg=cfg)
    trainer.load_model(model_path,'cpu')
    progress_bar = tqdm(trainer.eval_data_loader, ncols=40)

    draw_cnt = 0
    data_cnt = 0
    for j, data in enumerate(progress_bar):
        for key in data.keys():
            if isinstance(data[key], torch.DoubleTensor):
                data[key] = data[key].float()
        for k in range(data_aug):

            pred_agent = trainer.inference(copy.deepcopy(data))

            draw(data["rest"][0][0], pred_agent,cnt=draw_cnt,
                 edge=data["bound"][0][0], save=True)
            draw_cnt+=1

            meta = copy.deepcopy(datas[j])
            sdc_info = sdc_[j]
            sdc_x, sdc_y, sdc_yaw, sdc_theta = sdc_info[0][0], sdc_info[0][1], sdc_info[0][2], sdc_info[0][3]
            # for i in range(64):
            # pred = pred_agent[0, idx]
            # end_mask = end_mask[idx]
            xy = pred_agent[:, :2]
            vxy = pred_agent[:, 2:4]
            yaw = pred_agent[:, 6:8]
            endp = pred_agent[:,8:10]
            yaw = torch.atan2(yaw[:, 0], yaw[:, 1])
            yaw_ = (-yaw + sdc_yaw)

            vxy =  transform_coord(vxy, -sdc_theta)
            xy = transform_coord(xy, -sdc_theta)
            endp = transform_coord(endp, -sdc_theta)
            xy[:, 0] += sdc_x
            xy[:, 1] += sdc_y
            endp[:,0] += sdc_x
            endp[:,1] += sdc_y
            #type = pred[:, 8:]

            cnt = 1
            tracks = {}
            for k, agent in meta['tracks'].items():
                if k == meta['sdc_index']:
                    sdc_track = meta['tracks'][k]
                    sdc_track['state'] = sdc_track['state'][[0,-1]]
                    tracks[k] = sdc_track
                    continue
                if cnt>=pred_agent.shape[0]: continue
                track = meta['tracks'][k]

                track['state'][0, :2] = xy[cnt]
                track['state'][:, 3:5] = [5.286,2.332]
                track['state'][0, 7:9] = vxy[cnt]
                track['state'][:, -1] = 1
                track['state'][0, 6] = yaw_[cnt]
                track['type'] = AgentType.VEHICLE
                track['state'][1,:2] = endp[cnt]
                track['state'] = track['state'][:2]
                tracks[k] = track
                cnt += 1
            meta['tracks'] = tracks
            path = os.path.join(pred_data_path, f'{data_cnt}.pkl')
            data_cnt+=1
            with open(path, "wb") as f:
                pickle.dump(meta, f)




if __name__ == '__main__':
    cfg_path = './cfg/debug.yaml'

    model_path = '/Users/fenglan/model_47.pt'
    scenario_path = '/Users/fenglan/9s_training/selected'

    tt_path = '/Users/fenglan/Downloads/waymo/scenes_tt'
    pred_data_path = '/Users/fenglan/Downloads/waymo/scenes_pred'

    # model_path = '/users/s1155136634/drivingforce/drivingforce/TrafficTranformer/exps/tt_100epoch_22_01_16-05_05_14/saved_models/model_47.pt'
    # scenario_path = '/data0/pengzh/9s_training'
    #
    # tt_path = '/data0/pengzh/data_tt'
    # pred_data_path = '/data0/pengzh/9s_generated'

    data_num = 20
    data_aug = 1
    load_for_metadrive(model_path,scenario_path,cfg_path,tt_path,pred_data_path,data_num,data_aug)
