import os
from TrafficGen_init.data_process.agent_process import WaymoAgent
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import Tensor
import copy
from utils.utils import process_map,rotate,cal_rel_dir
from shapely.geometry import Polygon

LANE_SAMPLE = 10
RANGE = 50


class initDataset(Dataset):
    """
    If in debug, it will load debug dataset
    """

    def __init__(self, cfg, args=None, eval=False):
        self.total_data_usage = cfg["data_usage"] if not eval else cfg["eval_data_usage"]
        self.data_path = cfg['data_path']
        self.rank = args.rank if args is not None else 1
        self.process_num = args.world_size if args is not None and args.distributed else 1
        self.in_debug = cfg['debug']
        #self.recache = cfg['recache']
        self.eval = eval
        self.data_len = None
        self.data_loaded = {}
        self.cfg = cfg
        self.pad_num = cfg['pad_num']
        self.load_data()

        super(initDataset, self).__init__()

    def load_data(self):

        data_usage_in_this_proc = self.total_data_usage if self.eval else int(self.total_data_usage / self.process_num)
        start_index = self.cfg["eval_data_start_index"] if self.eval else self.rank * data_usage_in_this_proc
        end_index = start_index + data_usage_in_this_proc
        cnt = 0
        file_cnt = 0

        while file_cnt+start_index < end_index:
            index = file_cnt+start_index
            data_path = self.data_path
            data_file_path = os.path.join(data_path, f'{index}.pkl')
            with open(data_file_path, 'rb+') as f:
                datas = pickle.load(f)
                file_cnt+=1
            data = self.process(datas)
            case_cnt=0
            data_len = 1 if self.eval else len(data)

            for i in range(data_len):
                agent_num = data[i]['agent_mask'].sum()
                if agent_num<self.cfg['min_agent']:
                    continue
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


    def filter_agent(self, case_info):

        thres = 5
        max_agent_num = 32
        # process future agent

        agent = case_info['agent']
        vectors = case_info["center"]

        agent_mask = case_info['agent_mask']

        vec_x = ((vectors[...,0]+vectors[...,2])/2)
        vec_y = ((vectors[...,1]+vectors[...,3])/2)

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]

        b,vec_num = vec_y.shape
        _,agent_num = agent_x.shape

        vec_x = np.repeat(vec_x[:,np.newaxis],axis=1,repeats=agent_num)
        vec_y = np.repeat(vec_y[:, np.newaxis], axis=1, repeats=agent_num)

        agent_x = np.repeat(agent_x[:,:,np.newaxis],axis=-1,repeats=vec_num)
        agent_y = np.repeat(agent_y[:,:,np.newaxis],axis=-1,repeats=vec_num)

        dist = np.sqrt((vec_x - agent_x) ** 2 + (vec_y - agent_y) ** 2)

        cent_mask = np.repeat(case_info['center_mask'][:,np.newaxis],axis=1,repeats=agent_num)
        dist[cent_mask==0] = 10e5
        vec_index = np.argmin(dist,-1)
        min_dist_to_lane = np.min(dist,-1)
        min_dist_mask = min_dist_to_lane < thres

        selected_vec = np.take_along_axis(vectors,vec_index[...,np.newaxis],axis=1)

        vx,vy = agent[...,2], agent[...,3]
        v_value = np.sqrt(vx**2+vy**2)
        low_vel = v_value<0.5

        dir_v = np.arctan2(vy,vx)
        x1,y1,x2,y2 = selected_vec[...,0],selected_vec[...,1],selected_vec[...,2],selected_vec[...,3]
        dir = np.arctan2(y2-y1,x2-x1)
        agent_dir = agent[...,4]

        v_relative_dir = cal_rel_dir(dir_v,agent_dir)
        relative_dir = cal_rel_dir(agent_dir,dir)

        v_relative_dir[low_vel] = 0

        v_dir_mask = abs(v_relative_dir)<np.pi/6
        dir_mask = abs(relative_dir)<np.pi/4

        agent_x = agent[...,0]
        agent_y = agent[...,1]
        vec_x = (x1+x2)/2
        vec_y = (y1+y2)/2

        cent_to_agent_x = agent_x - vec_x
        cent_to_agent_y = agent_y - vec_y

        coord = rotate(cent_to_agent_x,cent_to_agent_y,-dir)

        vec_len = np.clip(np.sqrt(np.square(y2-y1) + np.square(x1-x2)), a_min=4.5, a_max=5.5)

        lat_perc = np.clip(-coord[...,1],a_min=-vec_len/2,a_max=vec_len/2)/vec_len
        long_perc = np.clip(coord[...,0],a_min=-vec_len/2,a_max=vec_len/2)/vec_len

        total_mask = min_dist_mask*agent_mask*v_dir_mask*dir_mask
        total_mask[:,0]=1
        total_mask = total_mask.astype(bool)

        b_s,agent_num,agent_dim = agent.shape
        agent_ = np.zeros([b_s,max_agent_num,agent_dim])
        agent_mask_ = np.zeros([b_s,max_agent_num]).astype(bool)


        the_vec = np.take_along_axis(vectors,vec_index[...,np.newaxis],1)
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 velocity and direction
        # 6-9 lane vector
        # 10-11 lane type and traff state
        info = np.concatenate(
            [vec_index[...,np.newaxis], long_perc[...,np.newaxis], lat_perc[...,np.newaxis],
             v_value[...,np.newaxis], v_relative_dir[...,np.newaxis],relative_dir[...,np.newaxis],the_vec], -1)

        info_ = np.zeros([b_s,max_agent_num,info.shape[-1]])

        for i in range(agent.shape[0]):
            agent_i = agent[i][total_mask[i]]
            info_i = info[i][total_mask[i]]

            agent_i = agent_i[:max_agent_num]
            info_i = info_i[:max_agent_num]

            valid_num = agent_i.shape[0]
            agent_i = np.pad(agent_i,[[0,max_agent_num-agent_i.shape[0]],[0,0]])
            info_i = np.pad(info_i, [[0, max_agent_num - info_i.shape[0]], [0, 0]])

            agent_[i] = agent_i
            info_[i] = info_i
            agent_mask_[i,:valid_num] = True


        # case_info['vec_index'] = info[...,0].astype(int)
        # case_info['relative_dir'] = info[..., 1]
        # case_info['long_perc'] = info[..., 2]
        # case_info['lat_perc'] = info[..., 3]
        # case_info['v_value'] = info[..., 4]
        # case_info['v_dir'] = info[..., 5]

        case_info['vec_based_rep'] = info_[...,1:]
        case_info['agent_vec_indx'] = info_[...,0].astype(int)
        case_info['agent_mask'] = agent_mask_
        case_info["agent"] = agent_

        return


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
        return lane

    def process_agent(self,agent,sort_agent):

        ego = agent[:,0]

        ego_pos = copy.deepcopy(ego[:,:2])[:,np.newaxis]
        ego_heading = ego[:,[4]]

        agent[...,:2] -= ego_pos
        agent[..., :2] = rotate(agent[...,0],agent[...,1],-ego_heading)
        agent[...,2:4] = rotate(agent[...,2],agent[...,3],-ego_heading)
        agent[...,4]-=ego_heading

        agent_mask = agent[...,-1]
        agent_type_mask = agent[...,-2]
        agent_range_mask = (abs(agent[...,0])<RANGE)*(abs(agent[...,1])<RANGE)
        mask = agent_mask*agent_type_mask*agent_range_mask

        bs, agent_num,_ = agent.shape
        sorted_agent = np.zeros_like(agent)
        sorted_mask = np.zeros_like(agent_mask).astype(bool)
        sorted_agent[:,0] = agent[:,0]
        sorted_mask[:,0]=True
        for i in range(bs):
            xy = copy.deepcopy(agent[i,1:,:2])
            agent_i = copy.deepcopy(agent[i,1:])
            mask_i = mask[i,1:]

            # put invalid agent to the right down side
            xy[mask_i==False,0] = 10e8
            xy[mask_i==False,1] = -10e8

            raster = np.floor(xy/0.25)
            raster = np.concatenate([raster, agent_i,mask_i[:,np.newaxis]], -1)
            y_index = np.argsort(-raster[:, 1])
            raster = raster[y_index]
            y_set = np.unique(raster[:, 1])[::-1]
            for y in y_set:
                ind = np.argwhere(raster[:, 1] == y)[:, 0]
                ys = raster[ind]
                x_index = np.argsort(ys[:, 0])
                raster[ind] = ys[x_index]
            #scene = np.delete(raster, [0, 1], axis=-1)
            sorted_agent[i,1:]=raster[...,2:-1]
            sorted_mask[i,1:]=raster[...,-1]


        if sort_agent:
            return sorted_agent[..., :-1], sorted_mask
        else:
            agent_nums = np.sum(sorted_mask, axis=-1)
            for i in range(sorted_agent.shape[0]):
                agent_num = int(agent_nums[i])
                permut_idx = np.random.permutation(np.arange(1, agent_num)) - 1
                sorted_agent[i, 1:agent_num] = sorted_agent[i, 1:agent_num][permut_idx]
            return sorted_agent[..., :-1], sorted_mask


    def get_gt(self,case_info):
        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
        # 6-9 lane vector
        # 10-11 lane type and traff state
        center_num = case_info['center'].shape[1]
        lane_inp,agent_vec_indx,vec_based_rep,bbox =  \
            case_info['lane_inp'][:,:center_num], case_info['agent_vec_indx'], case_info['vec_based_rep'],case_info['agent'][...,5:7]
        b, lane_num, _ = lane_inp.shape
        gt_distribution = np.zeros([b, lane_num])
        gt_vec_based_coord = np.zeros([b, lane_num, 5])
        gt_bbox = np.zeros([b, lane_num, 2])
        for i in range(b):
            mask = case_info['agent_mask'][i].sum()
            indx = agent_vec_indx[i].astype(int)
            gt_distribution[i][indx[:mask]] = 1
            gt_vec_based_coord[i, indx] = vec_based_rep[i,:,:5]
            gt_bbox[i,indx] = bbox[i]
        case_info['gt_bbox'] = gt_bbox
        case_info['gt_distribution'] = gt_distribution
        case_info['gt_long_lat'] = gt_vec_based_coord[...,:2]
        case_info['gt_speed'] = gt_vec_based_coord[...,2]
        case_info['gt_vel_heading'] = gt_vec_based_coord[...,3]
        case_info['gt_heading'] = gt_vec_based_coord[..., 4]

    def _process_map_inp(self,case_info):
        center = copy.deepcopy(case_info['center'])
        center[..., :4] /= RANGE
        edge = copy.deepcopy(case_info['bound'])
        edge[..., :4] /= RANGE
        cross = copy.deepcopy(case_info['cross'])
        cross[..., :4] /= RANGE
        rest = copy.deepcopy(case_info['rest'])
        rest[..., :4] /= RANGE

        case_info['lane_inp'] = np.concatenate([center,edge,cross,rest],axis=1)
        case_info['lane_mask'] = np.concatenate([case_info['center_mask'],case_info['bound_mask'],case_info['cross_mask'],case_info['rest_mask']],axis=1)
        return

    def process(self, data):
        # if self.eval:
        #     other = {}
        #     agent = copy.deepcopy(data['all_agent'])
        #     data['all_agent'] = data['all_agent'][[0]]
        #     lane = self.transform_coordinate_map(data)
        #
        #     ego = agent[:, 0]
        #     ego_pos = copy.deepcopy(ego[[0], :2])[:, np.newaxis]
        #     ego_heading = ego[[0], [4]]
        #     agent[..., :2] -= ego_pos
        #     agent[..., :2] = rotate(agent[..., 0], agent[..., 1], -ego_heading)
        #     agent[..., 2:4] = rotate(agent[..., 2], agent[..., 3], -ego_heading)
        #     agent[..., 4] -= ego_heading
        #
        #     agent_mask = agent[..., -1]
        #     agent_type_mask = agent[..., -2]
        #     agent_range_mask = (abs(agent[..., 0]) < RANGE) * (abs(agent[..., 1]) < RANGE)
        #     mask = agent_mask * agent_type_mask * agent_range_mask
        #
        #     agent = WaymoAgent(agent)
        #
        #     other['gt_agent'] = agent.get_inp(act=True)
        #     other['gt_agent_mask'] = mask
        #     other['lane'] = lane[0]
        #     other['traf'] = data['traffic_light']
        #     other['unsampled_lane'] = data['unsampled_lane']
        #     return other

        case_info = {}
        gap = 20

        other = {}

        if self.eval:
            other['traf'] = data['traffic_light']
            other['unsampled_lane'] = data['unsampled_lane']


        agent = copy.deepcopy(data['all_agent'])
        data['all_agent'] = data['all_agent'][0:-1:gap]
        data['lane'] = self.transform_coordinate_map(data)
        data['traffic_light'] = data['traffic_light'][0:-1:gap]
        if self.eval:
            other['lane'] = data['lane'][0]
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


        if self.cfg['model']=='sceneGen':
            sort_agent = True
        else:
            sort_agent = False
        case_info["agent"],case_info["agent_mask"] = self.process_agent(data['all_agent'],sort_agent)
        case_info['center'],case_info['center_mask'],case_info['bound'], case_info['bound_mask'],\
        case_info['cross'],case_info['cross_mask'],case_info['rest'],case_info['rest_mask'] = process_map(data['lane'],data['traffic_light'],  lane_range=RANGE, offest=0)

        self.filter_agent(case_info)

        agent = WaymoAgent(case_info['agent'],case_info['vec_based_rep'])
        #
        case_info['agent_feat'] = agent.get_inp()

        self._process_map_inp(case_info)

        self.get_gt(case_info)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k,v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)

        if self.eval:
            case_list[0]['other'] = other


        return case_list


