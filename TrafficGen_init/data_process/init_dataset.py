import os
from TrafficGen_init.data_process.agent_process import WaymoScene
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import Tensor
import copy
from utils.utils import process_map,rotate,cal_rel_dir

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
        self.recache = cfg['recache']
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

        # if self.eval:
        #     while file_cnt + start_index < end_index:
        #         index = file_cnt + start_index
        #         data_path = self.data_path
        #         data_file_path = os.path.join(data_path, f'{index}.pkl')
        #         with open(data_file_path, 'rb+') as f:
        #             datas = pickle.load(f)
        #         data = self.process(datas)
        #         self.data_loaded[file_cnt] = data
        #         file_cnt += 1
        #     self.data_len = file_cnt
        # else:
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
                if agent_num<self.cfg['min_agent'] and not self.eval:
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
        #v_value[low_vel] = 0

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
        A = y2-y1
        B = x1-x2
        C = -x1*A-y1*B
        vec_len = np.sqrt(np.square(A) + np.square(B))
        lat_dist = np.abs(A*agent_x+B*agent_y+C)/vec_len
        lat_dist[np.isnan(lat_dist)] = 0

        side_dir = cal_rel_dir(np.arctan2(agent_y-y1,agent_x-x1),dir)
        lat_dist[side_dir>0] *= -1

        dist_to_start = np.square(agent_x-x1) + np.square(agent_y-y1)
        long_dist = np.sqrt(dist_to_start-np.square(lat_dist))

        long_perc = long_dist/vec_len
        long_perc[np.isnan(long_perc)]=0
        long_perc = np.clip(long_perc,a_min=0,a_max=1)
        lat_perc = lat_dist/thres
        lat_perc = np.clip(lat_perc, a_min=-1, a_max=1)


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

        case_info['vec_based_rep'] = info_
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

    def process_agent(self,data):

        agent = data['all_agent']
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

        return agent,mask

    def process_as_inp(self,data):
        max_speed = 30

        agent = copy.deepcopy(data['agent'])
        lane_inp = copy.deepcopy(data['center'])
        vec_based_rep = copy.deepcopy(data['vec_based_rep'])

        # normalize
        agent[...,:2]/=RANGE
        agent[..., 2:4]/=max_speed

        lane_inp[...,:4]/=RANGE
        vec_based_rep[...,6:10]/=RANGE
        vec_based_rep[..., 3]/=max_speed

        cos_head = np.cos(agent[...,4])
        sin_head = np.sin(agent[...,4])

        agent_feat = np.concatenate([agent[...,:4],cos_head[...,np.newaxis],sin_head[...,np.newaxis],agent[...,5:7],vec_based_rep[...,1:]],axis=-1)

        data['agent_feat'] = agent_feat
        data['lane_inp'] = lane_inp

        # 0: vec_index
        # 1-2 long and lat percent
        # 3-5 speed, angle between velocity and car heading, angle between car heading and lane vector
        # 6-9 lane vector
        # 10-11 lane type and traff state

        b, lane_num, _ = lane_inp.shape
        gt_distribution = np.zeros([b, lane_num])
        gt_vec_based_coord = np.zeros([b, lane_num, 5])
        for i in range(b):
            indx = vec_based_rep[i, :, 0].astype(int)
            gt_distribution[i][indx] = 1
            gt_vec_based_coord[i, indx] = vec_based_rep[i,:,1:6]


        data['gt_distribution'] = gt_distribution
        data['gt_vec_based_coord'] = gt_vec_based_coord

        return

    def process(self, data):
        # if self.eval:
        #     case_info = {}
        #     data['lane'] = self.transform_coordinate_map(data)
        #     case_info["agent"],fut, case_info["agent_mask"] = self.process_agent(data)
        #
        #     case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        #     case_info['cross'], case_info['cross_mask'], case_info['rest'] = self.process_map(data)
        #
        #     for k,v in case_info.items():
        #         if v.shape[0]==19:
        #             case_info[k] = v[[0]]
        #     self.filter_agent(case_info)
        #
        #     vec_ind = case_info['vec_index'].to(int)
        #     line_seg = Tensor(case_info['center'])
        #
        #     # agent info in world coord[:8] and vector coord[8:]
        #     padding_content = torch.cat([case_info['agent'], case_info['long_perc'].unsqueeze(-1),
        #                                  case_info['lat_perc'].unsqueeze(-1), case_info['relative_dir'].unsqueeze(-1),
        #                                  case_info['v_value'].unsqueeze(-1)], dim=-1)
        #
        #     gather_vec = vec_ind.view(*vec_ind.shape, 1).repeat(1, 1, 7)
        #     the_vec = torch.gather(line_seg, 1, gather_vec)
        #     line_with_agent = torch.cat([padding_content, the_vec], -1)
        #
        #     b, v, _ = line_seg.shape
        #     gt = torch.zeros([b, v, 5])
        #     padding_content = padding_content[..., 8:]
        #     agent_mask = case_info["agent_mask"]
        #     case_info["agent"] = case_info["agent"][case_info["agent_mask"].to(bool)].unsqueeze(0)
        #     line_with_agent = line_with_agent[case_info["agent_mask"].to(bool)].unsqueeze(0)
        #     agent_num = case_info["agent"].shape[1]
        #
        #     fut = fut[:agent_num]
        #     case_info['agent_mask'] = case_info['agent_mask'][:,:agent_num]
        #     pad_num=self.pad_num
        #     if agent_num<pad_num:
        #         pad = torch.zeros(1,pad_num-agent_num,19)
        #         line_with_agent = torch.cat([line_with_agent,pad],1)
        #         pad = torch.zeros(1,pad_num-agent_num)
        #         case_info["agent_mask"] = torch.cat([case_info["agent_mask"],pad],1)
        #
        #     for i in range(b):
        #         mask = agent_mask[i]
        #         agent_info = padding_content[i]
        #         indx = vec_ind[i]
        #         indx = indx[mask.to(bool)]
        #         gt[i, indx, 0] = 1
        #         gt[i, indx, 1:] = agent_info[mask.to(bool)]
        #
        #     case_info['gt'] = gt
        #     case_info['line_with_agent'] = line_with_agent
        #
        #
        #     # pos = data['sdc_pos'][0]
        #     # lane = data['unsampled_lane']
        #     # lane[..., :2] -= pos
        #     # sdc_theta = data['sdc_theta'][0]
        #     # x = lane[..., 0]
        #     # y = lane[..., 1]
        #     # x_transform = np.cos(sdc_theta) * x - np.sin(sdc_theta) * y
        #     # y_transform = np.cos(sdc_theta) * y + np.sin(sdc_theta) * x
        #     # output_coords = np.stack((x_transform, y_transform), axis=-1)
        #     # lane[..., :2] = output_coords
        #     #
        #     # other = {}
        #     # other['agent_traj'] = fut
        #     # other['lane'] = data['lane'][0]
        #     # other['traf'] = data['traf_p_c_f']
        #     # other['center_info'] = data['center_info']
        #     # other['unsampled_lane'] = lane
        #     # case_info['other'] = other
        #     return case_info
        #
        # else:

        case_info = {}
        gap = 20
        data['all_agent'] = data['all_agent'][0:-1:gap]
        data['traffic_light'] = data['traffic_light'][0:-1:gap]

        data['lane'] = self.transform_coordinate_map(data)

        case_info["agent"],case_info["agent_mask"] = self.process_agent(data)

        case_info['center'],case_info['center_mask'],case_info['bound'], case_info['bound_mask'],\
        case_info['cross'],case_info['cross_mask'],case_info['rest'],_ = process_map(data['lane'],data['traffic_light'],lane_range=RANGE, offest=0)

        self.filter_agent(case_info)

        self.process_as_inp(case_info)

        case_num = case_info['agent'].shape[0]
        case_list = []
        for i in range(case_num):
            dic = {}
            for k,v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)
        return case_list



    # def process_agent(self,data):
    #     case = {}
    #
    #     sdc_theta = data['sdc_theta']
    #     pos = data['sdc_pos']
    #     all_agent = np.concatenate([data['ego_p_c_f'][np.newaxis],data['nbrs_p_c_f']],axis=0)
    #     coord = self.rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
    #     vel = self.rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)
    #     yaw = -sdc_theta+np.pi/2
    #     all_agent[..., 4] = all_agent[..., 4] + yaw
    #     all_agent[..., :2] = coord
    #     all_agent[..., 2:4] = vel
    #
    #     pos = all_agent[0,0,:2]
    #
    #     rotate_theta = -all_agent[0,0,4]+np.pi/2
    #
    #     all_agent[...,:2] -= pos
    #
    #     coord = self.rotate(all_agent[..., 0], all_agent[..., 1], rotate_theta)
    #     vel = self.rotate(all_agent[..., 2], all_agent[..., 3], rotate_theta)
    #     all_agent[..., :2] = coord
    #     all_agent[..., 2:4] = vel
    #     all_agent[..., 4] = all_agent[..., 4] - all_agent[0,0,4]
    #
    #     valid_mask = all_agent[:,0, -1] == 1.
    #     type_mask = all_agent[:,0, -2] == 1.
    #     range_mask = (abs(all_agent[:,0, 0]) < RANGE) * (abs(all_agent[:,0, 1]) < RANGE)
    #
    #     mask = valid_mask * type_mask*range_mask
    #     all_agent = all_agent[mask]
    #     all_agent[...,5] = 5.286
    #     all_agent[..., 6] = 2.332
    #
    #     agent_mask = np.ones([1,all_agent.shape[0]]).astype(bool)
    #
    #     current_agent = all_agent[:,0,:-1]
    #     headings = -current_agent[..., 4][..., np.newaxis]
    #     sin_h = np.sin(headings)
    #     cos_h = np.cos(headings)
    #     current_agent = np.concatenate([current_agent, sin_h, cos_h], -1)
    #     current_agent = np.delete(current_agent, [4, 7], axis=-1)
    #
    #     all_agent[..., 4] = all_agent[..., 4] + np.pi/2
    #
    #     return current_agent[np.newaxis],all_agent, agent_mask
