import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from TrafficGen_act.models.critertion import loss_v1

LANE_SAMPLE = 10
RANGE = 60
MAX_AGENT = 32


def process_map(data, center_num=128, edge_num=128, lane_range=60, offest=-40):
    lane = data['lane']

    traf = data['traf']

    lane_with_traf = np.zeros([*lane.shape[:-1], 5])
    lane_with_traf[..., :4] = lane

    lane_i_id = lane[:, -1]
    for a_traf in traf:
        lane_id = a_traf[0]
        state = a_traf[-2]
        lane_idx = np.where(lane_i_id == lane_id)
        lane_with_traf[lane_idx, -1] = state

    lane = lane_with_traf

    lane_type = lane[:, 2]
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

    cent, cent_mask = process_lane(lane[center_ind], center_num, lane_range, offest)
    bound, bound_mask = process_lane(lane[bound_ind], edge_num, lane_range, offest)
    cross, cross_mask = process_lane(lane[cross_ind], 64, lane_range, offest)
    rest, rest_mask = process_lane(lane[rest], center_num, lane_range, offest)

    return cent, cent_mask, bound, bound_mask, cross, cross_mask, rest

def process_case_to_input(case, agent_range=60, center_num=128, edge_num=128):
    inp = {}
    agent = case['agent'][..., :-1]
    headings = -agent[..., 4][..., np.newaxis]
    sin_h = np.sin(headings)
    cos_h = np.cos(headings)
    agent = np.concatenate([agent, sin_h, cos_h], -1)
    agent = np.delete(agent, [4, 7], axis=-1)

    range_mask = (abs(agent[:, 0]) < agent_range) * (abs(agent[:, 1] - 40) < agent_range)
    agent = agent[range_mask]

    agent = agent[:32]
    mask = np.ones(agent.shape[0])
    mask = mask[:32]
    agent = np.pad(agent, ([0, 32 - agent.shape[0]], [0, 0]))
    mask = np.pad(mask, ([0, 32 - mask.shape[0]]))
    inp['agent'] = agent
    inp['agent_mask'] = mask

    inp['center'], inp['center_mask'], inp['bound'], inp['bound_mask'], \
    inp['cross'], inp['cross_mask'], inp['rest'] = process_map(case, center_num, edge_num, agent_range)
    return inp

def process_lane(lane,  max_vec,lane_range,offset = -40):

    # dist = lane[..., 0]**2+lane[..., 1]**2
    # idx = np.argsort(dist)
    # lane = lane[idx]

    lane_point_mask = (abs(lane[..., 0]) < lane_range) * (abs(lane[..., 1] + offset) < lane_range)

    lane_id = np.unique(lane[...,-2]).astype(int)

    vec_list = []
    vec_mask_list = []
    for id in lane_id:
        id_set = lane[...,-2]==id
        points = lane[id_set]
        masks = lane_point_mask[id_set]

        vector = np.zeros([points.shape[0]-1,7])
        vector[..., 0:2] = points[:-1, :2]
        vector[..., 2:4] = points[1:, :2]
        # id
        vector[..., 4] = points[1:, 3]
        # type
        vector[..., 5] = points[1:, 2]
        # traffic light
        vector[..., 6] = points[1:, 4]
        vec_mask = masks[:-1]*masks[1:]
        vector[vec_mask==0]=0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list,axis=0) if vec_list else np.zeros([0,7])
    vector_mask = np.concatenate(vec_mask_list,axis=0) if vec_mask_list else np.zeros([0],dtype=bool)
    vector = vector[vector_mask]

    dist = vector[..., 0]**2+vector[..., 1]**2
    idx = np.argsort(dist)
    vector = vector[idx]
    vector_mask = np.ones(vector.shape[0])

    vector = vector[:max_vec]
    vector_mask = vector_mask[:max_vec]

    vector = np.pad(vector, ([0, max_vec - vector.shape[0]], [0, 0]))
    vector_mask = np.pad(vector_mask, ([0, max_vec - vector_mask.shape[0]]))

    return vector,vector_mask

class actDataset(Dataset):
    """
    If in debug, it will load debug dataset
    """

    def __init__(self, cfg, args=None, eval=False):
        self.total_data_usage = cfg["data_usage"] if not eval else cfg["eval_data_usage"]
        self.data_path = cfg['data_path']
        self.pred_len = cfg['pred_len']
        self.rank = args.rank if args is not None else 0
        self.process_num = args.world_size if args is not None and args.distributed else 1
        self.in_debug = cfg['debug']
        self.recache = cfg['recache']
        self.eval = eval
        self.data_len = None
        self.data_loaded = {}
        self.scene_data = {}
        self.cfg = cfg
        self.load_data()
        super(actDataset, self).__init__()

    def load_data(self):

        data_usage_in_this_proc = self.total_data_usage if self.eval else int(self.total_data_usage / self.process_num)
        start_index = self.cfg["eval_data_start_index"] if self.eval else self.rank * data_usage_in_this_proc
        end_index = start_index + data_usage_in_this_proc
        cnt = 0
        file_cnt = 0


        if self.eval:
            while file_cnt+start_index < end_index and self.eval:
            #for data_id in [6]:
                case_path = self.data_path
                case_file_path = os.path.join(case_path, f'{file_cnt+start_index}.pkl')
                #case_file_path = os.path.join(case_path, f'{data_id}.pkl')
                with open(case_file_path, 'rb+') as f:
                    case = pickle.load(f)
                # self.scene_data[file_cnt] = self.process_scene(case)
                self.scene_data[file_cnt] = case
                file_cnt+=1
            self.data_len = file_cnt

        else:
            while file_cnt+start_index < end_index:
                index = file_cnt+start_index
                data_path = self.data_path

                data_file_path = os.path.join(data_path, f'{index}.pkl')

                with open(data_file_path, 'rb+') as f:
                    datas = pickle.load(f)
                data = self.process(copy.deepcopy(datas))

                file_cnt+=1
                case_cnt=0
                for i in range(len(data)):
                    self.data_loaded[cnt+case_cnt] = data[i]
                    case_cnt+=1
                cnt+=case_cnt
            self.data_len = cnt
        print('Dataset len: {} (rank: {}), start_index: {}, end_index: {}'.format(self.data_len, self.rank,
                                                                                       start_index, end_index))
    def __len__(self):
        # debug set length=478
        return self.data_len

    def __getitem__(self, index):
        """
        Calculate for saving spaces
        """
        return self.data_loaded[index]

    def process_scene(self,data):

        case = {}

        sdc_theta = data['sdc_theta']
        pos = data['sdc_pos']
        all_agent = np.concatenate([data['ego_p_c_f'][np.newaxis],data['nbrs_p_c_f']],axis=0)
        coord = self.rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
        vel = self.rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)
        yaw = -sdc_theta+np.pi/2
        all_agent[..., 4] = all_agent[..., 4] + yaw
        all_agent[..., :2] = coord
        all_agent[..., 2:4] = vel
        pred_list = np.append(np.array([1]),data['pred_list']).astype(bool)
        all_agent = all_agent[pred_list][:,0]

        valid_mask = all_agent[..., -1] == 1.
        type_mask = all_agent[:, -2] == 1.
        mask = valid_mask * type_mask
        all_agent = all_agent[mask]

        case['all_agent'] = all_agent
        case['lane'] = data['lane']
        case['traf'] = data['traf_p_c_f']
        return case


    def rotate(self,x,y,angle):
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        return output_coords

    def transform_coordinate_map(self,data):
        """
        Every frame is different
        """

        case_list = []
        timestep = data['ego_p_c_f'].shape[0]
        pred_list = data['pred_list']
        for i in range(0,timestep,self.pred_len):
            if i+self.pred_len>=timestep:
                break

            sdc_theta = data['sdc_theta'][i:i + self.pred_len]
            pos = data['sdc_pos'][i:i + self.pred_len]
            all_agent = np.concatenate([data['ego_p_c_f'][np.newaxis,i:i+self.pred_len],data['nbrs_p_c_f'][:,i:i+self.pred_len]],axis=0)
            coord = self.rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
            vel = self.rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)

            yaw = -sdc_theta+np.pi/2

            all_agent[..., 4] = all_agent[..., 4] + yaw
            all_agent[..., :2] = coord
            all_agent[..., 2:4] = vel

            for j in range(all_agent.shape[0]):
                agents = copy.deepcopy(all_agent)
                one_case = {}
                if np.any(agents[j,:,-1]==0):
                    continue
                if not agents[j,0,-2] == 1:
                    continue
                if j>0 and pred_list[j-1]==0:
                    continue

                pos0 = copy.deepcopy(agents[j][0,:2])
                theta0 = -(copy.deepcopy(agents[j][0, 4])-np.pi/2)

                coord = agents[...,:2]
                vel = agents[...,2:4]
                coord-=pos0

                coord = self.rotate(coord[...,0],coord[...,1],theta0)
                vel = self.rotate(vel[...,0],vel[...,1],theta0)

                agents[...,:2]=coord
                agents[...,2:4]=vel
                agents[...,4]=agents[...,4]-copy.deepcopy(agents[j][0, 4])

                # then recover lane's position
                lane = copy.deepcopy(data['lane'])
                lane[..., :2] -= pos0

                output_coords = self.rotate(lane[..., 0],lane[..., 1],theta0)

                lane[..., :2] = output_coords

                other_agent = np.delete(agents,j,axis=0)

                one_case['lane'] = lane
                one_case['ego'] = agents[j]
                one_case['other'] = other_agent

                one_case['traf'] = data['traf_p_c_f'][i]
                case_list.append(one_case)

        return case_list

    def process_agent(self,case):
        ego = case['ego'][np.newaxis]
        others = case['other']
        a_scene = np.concatenate([ego,others],axis=0)

        current_scene = a_scene[:,0]
        valid_mask = current_scene[..., -1] == 1.
        range_mask = (abs(current_scene[:, 0]) < RANGE) * (abs(current_scene[:, 1]-40) < RANGE)
        type_mask = current_scene[:, -2] == 1.

        mask = valid_mask*range_mask * type_mask

        a_scene = a_scene[mask][..., :-1]
        current_scene = a_scene[:,0]
        ego_future_traj = a_scene[0,]
        #
        angle = ego_future_traj[:,4]
        ego_future_traj[angle>np.pi,4]-=2*np.pi
        ego_future_traj[angle < -np.pi,4] += 2 * np.pi
        #
        other_future_traj = a_scene[1:]

        # use cos,sin to represent heading
        headings = -current_scene[:, 4][..., np.newaxis]
        sin_h = np.sin(headings)
        cos_h = np.cos(headings)
        current_scene = np.concatenate([current_scene, sin_h, cos_h], -1)
        current_scene = np.delete(current_scene, [4, 7], axis=-1)

        current_scene = current_scene[:MAX_AGENT]
        other_future_traj = other_future_traj[:MAX_AGENT-1]
        valid_num = current_scene.shape[0]
        current_scene = np.pad(current_scene, ([0,MAX_AGENT - current_scene.shape[0]], [0, 0]))
        other_future_traj = np.pad(other_future_traj, ([0,MAX_AGENT - other_future_traj.shape[0]-1], [0, 0],[0, 0]))
        agent_mask = np.zeros(MAX_AGENT)
        agent_mask[:valid_num] = 1


        return ego_future_traj,other_future_traj, current_scene,agent_mask

    def process_gt(self, gt):

        ret = {}
        velo = (gt[1:,2:4]+gt[:-1,2:4])/2
        #heading = -(gt[1:,4]+gt[:-1,4])/2
        angular_velo = gt[1:,4]-gt[:-1,4]
        speed = (velo[:,0]**2 + velo[:,1]**2)**0.5*0.1
        angular_velo = -np.clip(angular_velo,a_min=-1,a_max=1)
        #speed = speed/30
        #angular_velo = angular_velo/(np.pi/)
        #ret = np.concatenate([speed,angular_velo],axis=-1)

        # heading = np.cumsum(angular_velo[:,0],axis=-1)
        # x = np.cumsum(speed[:,0]*np.sin(heading),axis=-1)
        # y = np.cumsum(speed[:,0]*np.cos(heading),axis=-1)

        ret['speed'] = speed.astype('float32')
        ret['heading'] = angular_velo.astype('float32')
        return ret

    def process(self, data):

        case_list = self.transform_coordinate_map(data)
        #case_info

        data_list = []
        for case in case_list:
            case_info = {}
            ego_gt, other_gt, agent, agent_mask = self.process_agent(case)

            case_info['agent'] = agent
            case_info['ego_gt'] = ego_gt

            case_info['processed_gt'] = self.process_gt(ego_gt)
            case_info['other_gt'] = other_gt
            case_info['agent_mask'] = agent_mask

            case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
            case_info['cross'], case_info['cross_mask'], case_info['rest'] = process_map(case)

            data_list.append(case_info)

        return data_list


