import os
from TrafficGen_init.data_process.agent_process import WaymoScene
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch
from torch import Tensor

LANE_SAMPLE = 10
RANGE = 50
case_num = 19

def process_lane(lane,  max_vec):

    lane_point_mask = (abs(lane[..., 0]) < RANGE) * (abs(lane[..., 1]) < RANGE)

    lane_id = np.unique(lane[...,-2]).astype(int)

    vec_list = []
    vec_mask_list = []
    for id in lane_id:
        id_set = lane[...,-2]==id
        points = lane[id_set].reshape(id_set.shape[0],-1,5)
        masks = lane_point_mask[id_set].reshape(id_set.shape[0],-1)

        vector = np.zeros([points.shape[0], points.shape[1]-1,7])
        vector[..., 0:2] = points[..., :-1, :2]
        vector[..., 2:4] = points[..., 1:, :2]
        # id
        vector[..., 4] = points[..., 1:, 3]
        # type
        vector[..., 5] = points[..., 1:, 2]
        # traffic light
        vector[..., 6] = points[..., 1:, 4]
        vec_mask = masks[:,:-1]*masks[:,1:]
        vector[vec_mask==0]=0
        vec_list.append(vector)
        vec_mask_list.append(vec_mask)

    vector = np.concatenate(vec_list,axis=1) if vec_list else np.zeros([19,0,7])
    vector_mask = np.concatenate(vec_mask_list,axis=1) if vec_mask_list else np.zeros([19,0],dtype=bool)

    vec_list = []
    vec_mask_list = []
    for i in range(vector.shape[0]):
        vec = vector[i][vector_mask[i]]

        dist = vec[..., 0] ** 2 + vec[..., 1] ** 2
        idx = np.argsort(dist)
        vec = vec[idx]
        vec_mask = np.ones(vec.shape[0])

        vec = vec[:max_vec]
        vec_mask = vec_mask[:max_vec]

        vec = np.pad(vec, ([0, max_vec - vec.shape[0]], [0, 0]))
        vec_mask = np.pad(vec_mask, ([0, max_vec - vec_mask.shape[0]]))
        vec_list.append(vec)
        vec_mask_list.append(vec_mask)


    vector = np.stack(vec_list)
    vector_mask = np.stack(vec_mask_list)


    return vector,vector_mask

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

        if self.eval:
            while file_cnt + start_index < end_index:
                index = file_cnt + start_index
                data_path = self.data_path
                data_file_path = os.path.join(data_path, f'{index}.pkl')
                with open(data_file_path, 'rb+') as f:
                    datas = pickle.load(f)
                data = self.process(datas)
                self.data_loaded[file_cnt] = data
                file_cnt += 1
            self.data_len = file_cnt
        else:
            while file_cnt+start_index < end_index:
                index = file_cnt+start_index
                data_path = self.data_path
                data_file_path = os.path.join(data_path, f'{index}.pkl')
                with open(data_file_path, 'rb+') as f:
                    datas = pickle.load(f)
                    file_cnt+=1
                data = self.process(datas)
                case_cnt=0
                for i in range(len(data)):
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
        # process future agent

        agent = Tensor(case_info['agent'])
        vectors = Tensor(case_info["center"])

        agent_mask = Tensor(case_info['agent_mask'])

        vec_x = ((vectors[...,0]+vectors[...,2])/2)
        vec_y = ((vectors[...,1]+vectors[...,3])/2)

        agent_x = agent[..., 0]
        agent_y = agent[..., 1]

        b,vec_num = vec_y.shape
        _,agent_num = agent_x.shape

        vec_x = vec_x.unsqueeze(1).repeat(1,agent_num,1)
        vec_y = vec_y.unsqueeze(1).repeat(1, agent_num, 1)

        agent_x = agent_x.unsqueeze(-1).repeat(1, 1, vec_num)
        agent_y = agent_y.unsqueeze(-1).repeat(1, 1, vec_num)

        dist = torch.sqrt((vec_x - agent_x) ** 2 + (vec_y - agent_y) ** 2)

        cent_mask = Tensor(case_info['center_mask']).unsqueeze(1).repeat(1,agent_num,1)
        dist[cent_mask==0] = 10e5
        vec_index = torch.argmin(dist,-1)
        min_dist_to_lane = torch.min(dist,-1)[0]
        min_dist_mask = min_dist_to_lane < thres

        gather_vec = vec_index.unsqueeze(-1).repeat(1,1,6)
        selected_vec = torch.gather(vectors,index = gather_vec,dim=1,)


        vx,vy = agent[...,2], agent[...,3]
        v_value = torch.sqrt(vx**2+vy**2)

        low_vel = v_value<0.1
        v_value[low_vel] = 0

        dir_v = -torch.atan2(vy,vx)+np.pi/2
        x1,y1,x2,y2 = selected_vec[...,0],selected_vec[...,1],selected_vec[...,2],selected_vec[...,3]
        dir = -torch.atan2(y2-y1,x2-x1)+np.pi/2
        agent_dir = torch.atan2(Tensor(agent[...,6]),Tensor(agent[...,7]))
        v_relative_dir = dir_v - agent_dir
        relative_dir = agent_dir-dir
        relative_dir[relative_dir < -np.pi] += 2 * np.pi
        relative_dir[relative_dir > np.pi] -= 2 * np.pi
        v_relative_dir[v_relative_dir < -np.pi] += 2 * np.pi
        v_relative_dir[v_relative_dir > np.pi] -= 2 * np.pi
        v_relative_dir[low_vel] = 0

        v_dir_mask = torch.abs(v_relative_dir)<np.pi/6
        dir_mask = torch.abs(relative_dir)<np.pi/4

        agent_x = agent[...,0]
        agent_y = agent[...,1]
        A = y2-y1
        B = x1-x2
        C = -x1*A-y1*B
        vec_len = torch.sqrt(torch.square(A) + torch.square(B))
        lat_dist = torch.abs(A*agent_x+B*agent_y+C)/vec_len
        lat_dist[torch.isnan(lat_dist)] = 0

        side_dir = -torch.atan2(agent_y-y1,agent_x-x1)+np.pi/2-dir
        side_dir[side_dir<-np.pi]+=2*np.pi
        side_dir[side_dir>np.pi] -= 2 * np.pi
        lat_dist[side_dir<0] *= -1

        dist_to_start = torch.square(agent_x-x1) + torch.square(agent_y-y1)
        long_dist = torch.sqrt(dist_to_start-torch.square(lat_dist))

        long_perc = long_dist/vec_len
        long_perc[torch.isnan(long_perc)]=0
        long_perc = torch.clip(long_perc,min=0,max=1)
        lat_perc = lat_dist/thres
        lat_perc = torch.clip(lat_perc, min=-1, max=1)


        total_mask = min_dist_mask*agent_mask*v_dir_mask*dir_mask
        total_mask[:,0]=1
        a = torch.ones_like(total_mask).cumsum(-1)-1
        a = a*total_mask
        a[total_mask==0]=a.shape[-1]-1
        a = a.sort(-1)[0]

        info = torch.cat([vec_index.unsqueeze(-1),relative_dir.unsqueeze(-1),long_perc.unsqueeze(-1),lat_perc.unsqueeze(-1), v_value.unsqueeze(-1),v_relative_dir.unsqueeze(-1)],-1)

        valid_index = a.unsqueeze(-1).repeat(1,1,info.shape[-1]).to(int)
        info = torch.gather(info,1,valid_index)

        agent_index = a.unsqueeze(-1).repeat(1,1,agent.shape[-1]).to(int)
        agent = torch.gather(agent,1,agent_index)


        case_info['vec_index'] = info[...,0]
        case_info['relative_dir'] = info[..., 1]
        case_info['long_perc'] = info[..., 2]
        case_info['lat_perc'] = info[..., 3]
        case_info['v_value'] = info[..., 4]
        case_info['v_dir'] = info[..., 5]

        case_info['agent_mask'] = torch.abs(torch.sort(-total_mask)[0])
        case_info["agent"] = agent

        return


    @staticmethod
    def process_map(data):

        lane = data['lane']

        traf = data['traf_p_c_f']

        lane_with_traf = np.zeros([*lane.shape[:-1],5])
        lane_with_traf[...,:4] = lane

        for i in range(lane.shape[0]):
            lane_i_id = lane[i, :, -1]
            for a_traf in traf[i]:
                lane_id = a_traf[0]
                state = a_traf[-2]
                lane_idx = np.where(lane_i_id == lane_id)
                lane_with_traf[i,lane_idx,-1] = state


        lane = lane_with_traf

        ind = list(range(0,190,10))

        lane = lane[ind]

        lane_type = lane[0,:,2]
        center_1 = lane_type==1
        center_2 = lane_type==2
        center_3 = lane_type==3
        center_ind = center_1+center_2+center_3

        boundary_1 = lane_type==15
        boundary_2 = lane_type == 16
        bound_ind = boundary_1 + boundary_2

        cross_walk =lane_type==18
        speed_bump = lane_type==19
        cross_ind = cross_walk+speed_bump

        rest = ~(center_ind + bound_ind)


        cent, cent_mask = process_lane(lane[:, center_ind], 512)
        bound, bound_mask = process_lane(lane[:, bound_ind],256)
        cross, cross_mask = process_lane(lane[:, cross_ind], 64)
        rest,rest_mask = process_lane(lane[:, rest], 256)


        return cent,cent_mask,bound,bound_mask, cross,cross_mask,rest

    def transform_coordinate_map(self,data):
        """
        Every frame is different
        """
        timestep = data['ego_p_c_f'].shape[0]

        sdc_theta = data['sdc_theta'][:,np.newaxis]
        pos = data['sdc_pos'][:,np.newaxis]
        lane = data['lane'][np.newaxis]
        lane = np.repeat(lane,timestep,axis=0)
        lane[...,:2] -= pos

        x = lane[..., 0]
        y = lane[..., 1]
        x_transform = np.cos(sdc_theta) * x - np.sin(sdc_theta) * y
        y_transform = np.cos(sdc_theta) * y + np.sin(sdc_theta) * x
        output_coords = np.stack((x_transform, y_transform), axis=-1)
        lane[...,:2] = output_coords

        return lane

    def rotate(self,x,y,angle):
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        return output_coords

    def process_agent(self,data):
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

        pos = all_agent[0,0,:2]

        rotate_theta = -all_agent[0,0,4]+np.pi/2

        all_agent[...,:2] -= pos

        coord = self.rotate(all_agent[..., 0], all_agent[..., 1], rotate_theta)
        vel = self.rotate(all_agent[..., 2], all_agent[..., 3], rotate_theta)
        all_agent[..., :2] = coord
        all_agent[..., 2:4] = vel
        all_agent[..., 4] = all_agent[..., 4] - all_agent[0,0,4]

        valid_mask = all_agent[:,0, -1] == 1.
        type_mask = all_agent[:,0, -2] == 1.
        range_mask = (abs(all_agent[:,0, 0]) < RANGE) * (abs(all_agent[:,0, 1]) < RANGE)

        mask = valid_mask * type_mask*range_mask
        all_agent = all_agent[mask]
        all_agent[...,5] = 5.286
        all_agent[..., 6] = 2.332

        agent_mask = np.ones([1,all_agent.shape[0]]).astype(bool)

        current_agent = all_agent[:,0,:-1]
        headings = -current_agent[..., 4][..., np.newaxis]
        sin_h = np.sin(headings)
        cos_h = np.cos(headings)
        current_agent = np.concatenate([current_agent, sin_h, cos_h], -1)
        current_agent = np.delete(current_agent, [4, 7], axis=-1)

        all_agent[..., 4] = all_agent[..., 4] + np.pi/2

        return current_agent[np.newaxis],all_agent, agent_mask

    def process(self, data):
        if self.eval:
            case_info = {}
            data['lane'] = self.transform_coordinate_map(data)
            case_info["agent"],fut, case_info["agent_mask"] = self.process_agent(data)

            case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
            case_info['cross'], case_info['cross_mask'], case_info['rest'] = self.process_map(data)

            for k,v in case_info.items():
                if v.shape[0]==19:
                    case_info[k] = v[[0]]
            self.filter_agent(case_info)

            vec_ind = case_info['vec_index'].to(int)
            line_seg = Tensor(case_info['center'])

            # agent info in world coord[:8] and vector coord[8:]
            padding_content = torch.cat([case_info['agent'], case_info['long_perc'].unsqueeze(-1),
                                         case_info['lat_perc'].unsqueeze(-1), case_info['relative_dir'].unsqueeze(-1),
                                         case_info['v_value'].unsqueeze(-1)], dim=-1)

            gather_vec = vec_ind.view(*vec_ind.shape, 1).repeat(1, 1, 7)
            the_vec = torch.gather(line_seg, 1, gather_vec)
            line_with_agent = torch.cat([padding_content, the_vec], -1)

            b, v, _ = line_seg.shape
            gt = torch.zeros([b, v, 5])
            padding_content = padding_content[..., 8:]
            agent_mask = case_info["agent_mask"]
            case_info["agent"] = case_info["agent"][case_info["agent_mask"].to(bool)].unsqueeze(0)
            line_with_agent = line_with_agent[case_info["agent_mask"].to(bool)].unsqueeze(0)
            agent_num = case_info["agent"].shape[1]

            fut = fut[:agent_num]
            case_info['agent_mask'] = case_info['agent_mask'][:,:agent_num]
            pad_num=self.pad_num
            if agent_num<pad_num:
                pad = torch.zeros(1,pad_num-agent_num,19)
                line_with_agent = torch.cat([line_with_agent,pad],1)
                pad = torch.zeros(1,pad_num-agent_num)
                case_info["agent_mask"] = torch.cat([case_info["agent_mask"],pad],1)

            for i in range(b):
                mask = agent_mask[i]
                agent_info = padding_content[i]
                indx = vec_ind[i]
                indx = indx[mask.to(bool)]
                gt[i, indx, 0] = 1
                gt[i, indx, 1:] = agent_info[mask.to(bool)]

            case_info['gt'] = gt
            case_info['line_with_agent'] = line_with_agent


            pos = data['sdc_pos'][0]
            lane = data['unsampled_lane']
            lane[..., :2] -= pos
            sdc_theta = data['sdc_theta'][0]
            x = lane[..., 0]
            y = lane[..., 1]
            x_transform = np.cos(sdc_theta) * x - np.sin(sdc_theta) * y
            y_transform = np.cos(sdc_theta) * y + np.sin(sdc_theta) * x
            output_coords = np.stack((x_transform, y_transform), axis=-1)
            lane[..., :2] = output_coords

            other = {}
            other['agent_traj'] = fut
            other['lane'] = data['lane'][0]
            other['traf'] = data['traf_p_c_f']
            other['center_info'] = data['center_info']
            other['unsampled_lane'] = lane
            case_info['other'] = other
            return case_info

        else:

            case_info = {}

            data['lane'] = self.transform_coordinate_map(data)
            #case_info

            scene = WaymoScene(data)
            case_info["agent"] = scene.agent_data
            case_info['agent'][...,4] = 5.286
            case_info['agent'][..., 5] = 2.332
            case_info["agent_mask"] = scene.agent_mask

            case_info['center'],case_info['center_mask'],case_info['bound'],case_info['bound_mask'],\
            case_info['cross'],case_info['cross_mask'],case_info['rest'] = self.process_map(data)


            self.filter_agent(case_info)

            vec_ind = case_info['vec_index'].to(int)
            line_seg = Tensor(case_info['center'])

            # agent info in world coord[:8] and vector coord[8:]
            padding_content = torch.cat([case_info['agent'],case_info['long_perc'].unsqueeze(-1),
                                         case_info['lat_perc'].unsqueeze(-1),case_info['relative_dir'].unsqueeze(-1),
                                         case_info['v_value'].unsqueeze(-1)],dim=-1)

            gather_vec = vec_ind.view(*vec_ind.shape,1).repeat(1,1,7)
            the_vec = torch.gather(line_seg,1,gather_vec)
            line_with_agent = torch.cat([padding_content,the_vec],-1)


            b,v,_ = line_seg.shape
            gt = torch.zeros([b, v, 5])
            padding_content = padding_content[..., 8:]
            agent_mask = case_info["agent_mask"]
            for i in range(b):
                mask =  agent_mask[i]
                agent_info = padding_content[i]
                indx = vec_ind[i]
                indx = indx[mask.to(bool)]
                gt[i,indx,0] = 1
                gt[i,indx,1:] = agent_info[mask.to(bool)]

            case_info['gt'] = gt
            case_info['line_with_agent'] = line_with_agent

            case_list = []
            for i in range(case_num):
                dic = {}
                for k,v in case_info.items():
                    dic[k] = v[i]
                case_list.append(dic)

            return case_list


