import copy
import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from utils.utils import process_map,rotate,cal_rel_dir
from TrafficGen_init.data_process.agent_process import WaymoAgent

LANE_SAMPLE = 10
RANGE = 60
MAX_AGENT = 32

def process_case_to_input(case, agent_range=60):
    inp = {}
    agent = WaymoAgent(case['agent'])
    agent = agent.get_inp(act_inp=True)
    range_mask = (abs(agent[:, 0]- 40/50) < agent_range/50) * (abs(agent[:, 1] ) < agent_range/50)
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
        # if self.eval:
        #     while file_cnt+start_index < end_index and self.eval:
        #     #for data_id in [6]:
        #         case_path = self.data_path
        #         case_file_path = os.path.join(case_path, f'{file_cnt+start_index}.pkl')
        #         #case_file_path = os.path.join(case_path, f'{data_id}.pkl')
        #         with open(case_file_path, 'rb+') as f:
        #             case = pickle.load(f)
        #         # self.scene_data[file_cnt] = self.process_scene(case)
        #         self.data_loaded[file_cnt] = case
        #         file_cnt+=1
        #     self.data_len = file_cnt
        # else:
        while cnt+start_index < end_index:
            index = cnt+start_index
            data_file_path = os.path.join(self.data_path, f'{index}.pkl')

            with open(data_file_path, 'rb+') as f:
                datas = pickle.load(f)
            self.data_loaded[cnt] = self.process(copy.deepcopy(datas))
            cnt+=1
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


    def process_agent(self,data):

        agent = data['all_agent']
        ego = agent[:,0]

        ego_pos = copy.deepcopy(ego[[0],:2])[:,np.newaxis]
        ego_heading = ego[[0],[4]]

        agent[...,:2] -= ego_pos
        agent[..., :2] = rotate(agent[...,0],agent[...,1],-ego_heading)
        agent[...,2:4] = rotate(agent[...,2],agent[...,3],-ego_heading)
        agent[...,4]-=ego_heading

        agent_mask = agent[...,-1]
        agent_type_mask = agent[...,-2]
        agent_range_mask = (abs(agent[...,0]-40)<RANGE)*(abs(agent[...,1])<RANGE)
        mask = agent_mask*agent_type_mask*agent_range_mask

        return agent, mask.astype(bool)


    def get_inp_gt(self,case_info,agent,agent_mask):
        agent_context = agent[0]
        agent_mask = agent_mask[0]
        agent_context = agent_context[agent_mask]
        agent_context = agent_context[:MAX_AGENT]
        agent_mask = agent_mask[:MAX_AGENT]
        agent_context = WaymoAgent(agent_context)
        agent_context = agent_context.get_inp(act_inp=True)
        agent_context = np.pad(agent_context,([0,MAX_AGENT-agent_context.shape[0]],[0,0]))
        agent_mask = np.pad(agent_mask,([0,MAX_AGENT-agent_mask.shape[0]]))

        case_info['agent'] = agent_context
        case_info['agent_mask'] = agent_mask

        ego_future = agent[:self.pred_len,0]

        case_info['gt_pos'] = ego_future[1:,:2]-ego_future[:-1,:2]
        case_info['gt_vel'] = ego_future[1:,2:4]
        case_info['gt_heading'] = cal_rel_dir(ego_future[1:,4],0)

        return agent_context, agent_mask,agent[:self.pred_len,0]

    def transform_coordinate_map(self,data):
        """
        Every frame is different
        """
        timestep = data['all_agent'].shape[0]

        #sdc_theta = data['sdc_theta'][:,np.newaxis]
        ego = data['all_agent'][:,0]
        pos = ego[:,[0,1]][:,np.newaxis]

        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane,timestep,axis=0)
        lane[...,:2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        ego_heading = ego[:,[4]]
        lane[...,:2] = rotate(x,y,-ego_heading)

        data['lane'] = lane
    def process(self, data):
        case_info = {}

        self.transform_coordinate_map(data)
        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'],case_info['rest_mask'] = process_map(data['lane'][[0]],[data['traffic_light'][0]],center_num=256,edge_num=128,offest=-40,lane_range=60)

        case_info['center'] = case_info['center'][0]
        case_info['center_mask'] = case_info['center_mask'][0]
        case_info['bound'] = case_info['bound'][0]
        case_info['bound_mask'] = case_info['bound_mask'][0]
        case_info['cross'] = case_info['cross'][0]
        case_info['cross_mask'] = case_info['cross_mask'][0]
        case_info['rest'] = case_info['rest'][0]
        case_info['rest_mask'] = case_info['rest_mask'][0]

        agent, agent_mask = self.process_agent(data)
        self.get_inp_gt(case_info, agent,agent_mask)

        return case_info


