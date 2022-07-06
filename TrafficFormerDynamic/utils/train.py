import datetime
import copy
import numpy as np
from shapely.geometry import Polygon

from drivingforce.TrafficFormerDynamic.utils.visualization import draw_heatmap,draw_seq
from drivingforce.TrafficFormerV2.utils.visualization import draw
from drivingforce.TrafficFormerDynamic.models.tt_v2 import TrafficFormerDynamic
from drivingforce.TrafficFormerDynamic.utils.visualization import get_agent_pos_from_vec
from drivingforce.TrafficFormerDynamic.utils.dataLoader import WaymoDataset
from drivingforce.TrafficFormerDynamic.load_for_metadrive import AgentType,RoadEdgeType,RoadLineType

from random import choices
import time

import torch
from torch import Tensor
import torch.distributed as dist
import wandb
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        ret = fn(*args, **kwargs)
        return ret, time.clock() - start

    return _wrapper


def get_time_str():
    return datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")



class Trainer:
    def __init__(self,
                 save_freq=1,
                 exp_name=None,
                 wandb_log_freq=100,
                 cfg=None,
                 args=None
                 ):
        self.args = args
        self.cfg = cfg

        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", rank=args.local_rank)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
            print('[TORCH] Training in distributed mode. Process %d, local %d, total %d.' % (
                args.rank, args.local_rank, args.world_size))

            # ===== load data =====
            train_dataset = WaymoDataset(self.cfg, args)
            data_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=cfg['batch_size'],
                                                      shuffle=True,
                                                      num_workers=self.cfg['num_workers'])
            model = TrafficFormerDynamic()
            model.cuda(args.local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            broadcast_buffers=False)  # only single machine
        else:
            # ===== load data =====
            model = TrafficFormerDynamic()
            model = torch.nn.DataParallel(model, list(range(1)))
            train_dataset = WaymoDataset(self.cfg, args)
            data_loader = DataLoader(train_dataset, shuffle=False, batch_size=cfg['batch_size'],
                                     num_workers=self.cfg['num_workers'])

            model = model.to(self.cfg['device'])

        optimizer = optim.AdamW(model.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                                weight_decay=self.cfg['weight_decay'], amsgrad=True)

        self.eval_data_loader = None
        if self.main_process and self.cfg["need_eval"]:
            test_set = WaymoDataset(self.cfg, args, eval=True)
            self.eval_data_loader = DataLoader(test_set, shuffle=False, batch_size=cfg['eval_batch_size'],
                                               num_workers=self.cfg['num_workers'])
        self.model = model
        self.train_dataloader = data_loader
        self.in_debug = cfg["debug"]  # if in debug, wandb will log to TEST instead of cvpr
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,30,35], gamma=0.2,verbose=True)
        self.batch_size = cfg['batch_size']
        self.max_epoch = cfg['max_epoch']
        self.current_epoch = 0
        self.save_freq = save_freq
        self.exp_name = "v2_{}_{}".format(exp_name, get_time_str()) if exp_name is not None else "dynamic_{}".format(
            get_time_str())
        self.exp_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exps",
                                          self.exp_name)
        self.wandb_log_freq = wandb_log_freq
        self.total_sgd_count = 1
        self.training_start_time = time.time()
        #self.mmd_loss = MMD_loss()
        if self.main_process:
            self.make_dir()
            # wandb.login(key="a6ae178e5596edd2aa7e54d4d34abebe5759406e")
            # wandb.init(
            #     entity="drivingforce",
            #     project="cvpr" if not self.in_debug else "TEST",
            #     name=self.exp_name,
            #     config=cfg)
            gpu_num = torch.cuda.device_count()
            print("gpu number:{}".format(gpu_num))
            print("gpu available:", torch.cuda.is_available())
            # wandb.watch(model) # gradient stat

    def train(self):
        while self.current_epoch < self.max_epoch:
            epoch_start_time = time.time()
            current_sgd_count = self.total_sgd_count
            train_data = self.load_data()
            _, epoch_training_time = self.train_one_epoch(train_data)
            self.scheduler.step()
            # log data in epoch level
            if self.main_process:

                wandb.log({
                    "epoch_training_time (s)": time.time() - epoch_start_time,
                    "sgd_iters_in_this_epoch": self.total_sgd_count - current_sgd_count,
                    "epoch": self.current_epoch,
                    "epoch_training_time": epoch_training_time,
                })

                if self.cfg["need_eval"] and self.current_epoch % self.cfg["eval_frequency"] == 0:
                    #self.eval_model()
                    self.save_model()



    def get_type_class(self,line_type):
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

    def generate_case_for_metadrive(self,data_path):


        output_temp = {}
        output_temp['id'] = 'fake'
        output_temp['ts'] = [x/10 for x in range(190)]
        output_temp['dynamic_map_states'] = [{}]
        output_temp['sdc_index'] = 0

        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader

            scene_data = eval_data.dataset.scene_data
            pred_list = []
            for k,v in tqdm(scene_data.items()):
                pred_list.append(self.inference(v))


            for i in range(len(pred_list)):
                pred = pred_list[i]['pred']
                agent_lerp = np.zeros([185,*pred.shape[1:]])
                for j in range(pred.shape[0]-1):
                    pred_j = pred[j,:,:4]
                    pred_j1 = pred[j+1,:,:4]
                    delta = pred_j1-pred_j
                    dt = 1/5
                    for k in range(5):
                        agent_lerp[j*5+k,:,:4] = pred_j+dt*k*delta
                        agent_lerp[j*5 + k, :, 4:] = pred[j,:,4:]
                pred_list[i]['pred'] = agent_lerp

            for i in range(len(pred_list)):
                gt_agent = scene_data[i]['gt_agent']
                #
                # invalid = [1,2,7]
                # pred_list[i]['pred'] = np.delete(pred_list[i]['pred'],invalid,axis=1)

                other = scene_data[i]['other']
                lane = pred_list[i]['lane']
                traf = other['traf']
                agent = pred_list[i]['pred']
                # delete invalid agent

                dir_path = f'./gifs/{i}'
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                #for t in range(agent.shape[0]):
                for t in [0,20,40,60,80,100,120,140,160,180]:
                    path = os.path.join(dir_path,f'{t}')
                    agent_t = agent[t]
                    traf_t = traf[t]
                    inp = {}
                    inp['traf'] = traf_t
                    inp['lane'] = lane
                    cent,cent_mask,bound,bound_mask,_,_,rest = WaymoDataset.process_map(inp,2000,1000,100)
                    if t==0:
                        draw_seq(cent,bound,rest,gt_agent,path=os.path.join(dir_path,'gt'),save=True,gt=True)
                    draw_seq(cent,bound,rest,agent_t,path=path,save=True)

            cnt=0
            for j in tqdm(range(len(pred_list))):
                case = pred_list[j]
                center_info = scene_data[j]['other']['center_info']
                output = copy.deepcopy(output_temp)
                output['tracks']={}
                output['map'] = {}
                # extract agents
                agent = case['pred']

                for i in range(agent.shape[1]):
                    track = {}
                    agent_i = agent[:,i]
                    track['type'] = AgentType.VEHICLE
                    state = np.zeros([agent_i.shape[0],10])
                    state[:,:2] = agent_i[:,:2]
                    state[:, 3] = 5.286
                    state[:, 4] = 2.332
                    state[:, 7:9] = agent_i[:,2:4]
                    state[:, -1] = 1
                    state[:, 6] = agent_i[:,4]+np.pi/2
                    track['state'] = state
                    output['tracks'][i] = track

                # extract maps
                lane = scene_data[j]['other']['unsampled_lane']
                lane_id = np.unique(lane[..., -1]).astype(int)
                for id in lane_id:

                    a_lane = {}
                    id_set = lane[..., -1] == id
                    points = lane[id_set]
                    polyline = np.zeros([points.shape[0],3])
                    line_type = points[0,-2]
                    polyline[:,:2] = points[:,:2]
                    a_lane['type'] = self.get_type_class(line_type)
                    a_lane['polyline'] = polyline
                    if id in center_info.keys():
                        a_lane.update(center_info[id])
                    output['map'][id] = a_lane

                p = os.path.join(data_path, f'{cnt}.pkl')
                with open(p, 'wb') as f:
                    pickle.dump(output, f)
                cnt+=1

    def save_case_for_metadrive(self,pred_list,scene_data,data_path):
        output_temp = {}
        output_temp['id'] = 'fake'
        output_temp['ts'] = [x/10 for x in range(190)]
        output_temp['dynamic_map_states'] = [{}]
        output_temp['sdc_index'] = 0

        cnt = 0
        for j in tqdm(range(len(pred_list))):
            case = pred_list[j]
            center_info = scene_data[j]['other']['center_info']
            output = copy.deepcopy(output_temp)
            output['tracks'] = {}
            output['map'] = {}
            # extract agents
            agent = case['pred']

            for i in range(agent.shape[1]):
                track = {}
                agent_i = agent[:, i]
                track['type'] = AgentType.VEHICLE
                state = np.zeros([agent_i.shape[0], 10])
                state[:, :2] = agent_i[:, :2]
                state[:, 3] = 5.286
                state[:, 4] = 2.332
                state[:, 7:9] = agent_i[:, 2:4]
                state[:, -1] = 1
                state[:, 6] = agent_i[:, 4] + np.pi / 2
                track['state'] = state
                output['tracks'][i] = track

            # extract maps
            lane = scene_data[j]['other']['unsampled_lane']
            lane_id = np.unique(lane[..., -1]).astype(int)
            for id in lane_id:

                a_lane = {}
                id_set = lane[..., -1] == id
                points = lane[id_set]
                polyline = np.zeros([points.shape[0], 3])
                line_type = points[0, -2]
                polyline[:, :2] = points[:, :2]
                a_lane['type'] = self.get_type_class(line_type)
                a_lane['polyline'] = polyline
                if id in center_info.keys():
                    a_lane.update(center_info[id])
                output['map'][id] = a_lane

            p = os.path.join(data_path, f'{cnt}.pkl')
            with open(p, 'wb') as f:
                pickle.dump(output, f)
            cnt += 1


    def draw_showcase(self):
        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader

            scene_data = eval_data.dataset.scene_data
            pred_list = []
            for k, v in tqdm(scene_data.items()):
                pred_list.append(self.inference(v))

            for i in range(len(pred_list)):
                data_i = scene_data[i]
                other = data_i['other']
                inp_lane = {}
                inp_lane['traf'] = other['traf'][0]
                inp_lane['lane'] = other['unsampled_lane']
                cent, cent_mask, bound, bound_mask, _, _, rest = WaymoDataset.process_map(inp_lane, 7000, 5000, 100, 0)

                draw(cent, pred_list[i]['pred'], edge=bound,rest=rest,
                     context=pred_list[i]['agent'])
                cnt = 0

                # other = scene_data[i]['other']
                # lane = pred_list[i]['lane']
                # traf = other['traf']
                # agent = pred_list[i]['pred']
                # dir_path = f'./gifs/{i}'
                # if not os.path.exists(dir_path):
                #     os.mkdir(dir_path)
                # for t in range(agent.shape[0]):
                #     path = os.path.join(dir_path, f'{t}')
                #     agent_t = agent[t]
                #     traf_t = traf[t * 5]
                #     inp = {}
                #     inp['traf'] = traf_t
                #     inp['lane'] = lane
                #     cent, cent_mask, bound, bound_mask, _, _, _ = WaymoDataset.process_map(inp, 2000, 1000, 100)
                #     draw_seq(cent, bound, agent_t, path, True)

    def draw_gifs(self):
        output_temp = {}
        output_temp['id'] = 'fake'
        output_temp['ts'] = [x / 10 for x in range(190)]
        output_temp['dynamic_map_states'] = [{}]
        output_temp['sdc_index'] = 0

        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader

            scene_data = eval_data.dataset.scene_data
            pred_list = []

            for i in tqdm(range(len(scene_data))):
                data_inp = scene_data[i]
                #heat_map = data_inp['heat_map']
                pred_i = self.inference_control(data_inp)
                pred_i['pred']=np.delete(pred_i['pred'],4,axis=1)
                pred_list.append(pred_i)

            #self.save_case_for_metadrive(pred_list,scene_data,'/Users/fenglan/Downloads/waymo/generated_metadrive')

                other = scene_data[i]['other']
                lane = pred_i['lane']
                traf = other['traf']
                agent = pred_i['pred']
                dir_path = f'./gifs/{i}'
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)

                # ind = list(range(0,190,5))
                # agent = np.delete(agent,4,axis=1)
                # #del heat_map[5]
                # agent = agent[ind]
                #for t in range(agent.shape[0]):
                for t in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]:
                    agent_t = agent[t]
                    #path = os.path.join(dir_path,f'{t+agent_t.shape[0]-1}')
                    path = os.path.join(dir_path, f'{t}')
                    traf_t = traf[t]
                    inp = {}
                    inp['traf'] = traf_t
                    inp['lane'] = lane
                    cent,cent_mask,bound,bound_mask,_,_,rest = WaymoDataset.process_map(inp, 2000, 1000, 70, 0)
                    #cent,cent_mask,bound,bound_mask,_,_,rest = WaymoDataset.process_map(inp,2000,1000,100)
                    # if t==0:
                    #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                    #     for k in range(1,agent_t.shape[0]):
                    #         heat_path = os.path.join(dir_path, f'{k-1}')
                    #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)
                    draw_seq(cent,bound,rest,agent_t,path=path,save=True)



    def eval_model(self):

        self.model.eval()
        ret = {}
        loss = 0
        losses = []

        with torch.no_grad():
            eval_data = self.eval_data_loader
            cnt = 0

            scene_data = eval_data.dataset.scene_data
            pred_list = []
            for k,v in scene_data.items():
                pred_list.append(self.inference(v))

            for i in range(len(pred_list)):
                plt_pred = wandb.Image(draw(pred_list[i]['center'],pred_list[i]['pred'] ,edge=pred_list[i]['bound'],context=pred_list[i]['agent']))
                ret[f"pred_{i}"] = plt_pred

            for batch in eval_data:
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()

                inp_batch = copy.deepcopy(batch)
                pred, _, loss_, losses_ = self.model(inp_batch)

                loss+=loss_
                losses.append(losses_)

                cnt += 1


        loss = loss/cnt
        dic = {}
        for i in range(len(losses)):
            lo = losses[i]
            for k,v in lo.items():
                if i == 0:
                    dic[k] = v
                else:
                    dic[k] += v
        for k,v in dic.items():
            dic[k] = (v/cnt)

        dic['loss'] = loss
        wandb.log({'eval':dic})
        wandb.log(ret)

    def get_polygon(self, center, yaw, L, W):
        # agent = WaymoAgentInfo(agent)
        #yaw = yaw.cpu()

        l, w = L / 2, W / 2
        theta = np.arctan(l / w)
        s1 = np.sqrt(l ** 2 + w ** 2)
        x1 = abs(np.cos(theta + yaw) * s1)
        y1 = abs(np.sin(theta + yaw) * s1)
        x2 = abs(np.cos(theta - yaw) * s1)
        y2 = abs(np.sin(theta - yaw) * s1)

        p1 = [center[0] + x1, center[1] - y1]
        p2 = [center[0] - x1, center[1] + y1]
        p3 = [center[0] + x2, center[1] + y2]
        p4 = [center[0] - x2, center[1] - y2]
        return Polygon([p1, p3, p2, p4])

    def rotate(self, x, y, angle):
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
        return output_coords

    # give the first agent and map, produce 64 agent

    def transform_to_agent(self,agent_i,agent,lane):

        all_ = copy.deepcopy(agent)

        center = copy.deepcopy(agent_i[:2])
        center_yaw = copy.deepcopy(agent_i[4])
        rotate_theta = -(center_yaw - np.pi / 2)

        all_[..., :2] -= center

        coord = self.rotate(all_[..., 0], all_[..., 1], rotate_theta)
        vel = self.rotate(all_[..., 2], all_[..., 3], rotate_theta)
        all_[..., :2] = coord
        all_[..., 2:4] = vel
        all_[..., 4] = all_[..., 4] - center_yaw
        # then recover lane's position
        lane = copy.deepcopy(lane)
        lane[..., :2] -= center
        output_coords = self.rotate(lane[..., 0], lane[..., 1], rotate_theta)
        lane[..., :2] = output_coords

        return all_,lane

    def process_case_to_input(self,case,agent_range=60, center_num=128,edge_num=128):
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
        inp['cross'], inp['cross_mask'], inp['rest'] = self.eval_data_loader.dataset.process_map(case,center_num,edge_num,agent_range)
        return inp

    def inference(self, data, length = 38, per_time = 6):

        # for every x time step, pred then update

        future_traj = data['other']['agent_traj']


        pred_agent = np.zeros([length,*data['all_agent'].shape])
        pred_agent[0] = copy.deepcopy(data['all_agent'])
        vel = (pred_agent[0,:,2]**2 + pred_agent[0,:,3]**2)**0.5
        pred_agent[0][vel<0.05,2:4]=0
        pred_agent[1:,:,5:] = pred_agent[0,:,5:]

        ind = list(range(0,190,5))
        pred_agent[:,0] = future_traj[0,ind]
        agent_num = data['all_agent'].shape[0]

        for i in range(0,length-1,per_time):

            current_agent = copy.deepcopy(pred_agent[i])
            case_list = []
            for j in range(agent_num):
                a_case = {}
                a_case['agent'],a_case['lane'] = self.transform_to_agent(current_agent[j],current_agent,data['lane'])
                a_case['traf'] = data['other']['traf'][i]
                case_list.append(a_case)

            inp_list = []
            for case in case_list:
                inp_list.append(self.process_case_to_input(case))

            keys = inp_list[0].keys()

            batch = {}
            for key in keys:
                one_item = [item[key] for item in inp_list]
                batch[key] = Tensor(np.stack(one_item))

            # for inp in inp_list:
            #     draw(inp['center'], inp['agent'][inp['agent_mask'].astype(bool)][np.newaxis], context=inp['agent'][inp['agent_mask'].astype(bool)])
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    batch[key] = batch[key].cuda()
            if self.cfg['device'] == 'cuda':
                self.model.cuda()
            pred, prob = self.model(batch,False)


            best_pred_idx = torch.argmax(prob,dim=-1)
            best_pred_idx = best_pred_idx.view(pred.shape[0],1,1,1).repeat(1,1,*pred.shape[2:])
            best_pred = torch.gather(pred,dim=1,index=best_pred_idx).squeeze(1).cpu().numpy()

            best_pred = np.concatenate([np.zeros([best_pred.shape[0],1, 2]), best_pred], axis=1)
            best_pred = best_pred[:,1:] - best_pred[:,:-1]
            buf = np.zeros([best_pred.shape[0],8,2])
            cnt = 0
            for z in range(0,40,5):
                slice = best_pred[:,z:z+5]
                buf[:,cnt] = slice.sum(-2)
                cnt+=1



            ## update all the agent
            for j in range(1,agent_num):
                pred_j = buf[j]
                agent_j = copy.deepcopy(current_agent[j])
                center = copy.deepcopy(agent_j[:2])
                center_yaw = copy.deepcopy(agent_j[4])
                rotate_theta = -(center_yaw-np.pi/2)

                # pred_j = np.concatenate([np.zeros([1,2]),pred_j],axis=0)
                # pred_j = pred_j[1:]-pred_j[:-1]
                pred_j = self.rotate(pred_j[:, 0], pred_j[:, 1], -rotate_theta)

                heading = np.arctan2(pred_j[..., 1], pred_j[..., 0])
                #jerk = np.zeros_like(heading).astype(bool)
                heading = np.concatenate([np.array([center_yaw]),heading])

                vel = (pred_j[:, 0] ** 2 + pred_j[:, 1] ** 2) ** 0.5

                # for k in range(1,len(heading)):
                #     vel_i = vel[k-1]
                #     heading_diff = heading[k]-heading[k-1]
                #     while heading_diff > np.pi:
                #         heading_diff-=2*np.pi
                #     while heading_diff < -np.pi:
                #         heading_diff+=2*np.pi
                #     if abs(heading_diff)>np.pi/6 or vel_i<0.06:
                #         heading[k] = heading[k-1]
                #         pred_j[k - 1] = 0
                    #if abs(heading_diff)>np.pi/4:


                pos = pred_j.cumsum(-2) + center
                vel = pred_j / 0.5

                pad_len = pred_agent[i+1:i+per_time+1].shape[0]
                pred_agent[i+1:i+per_time+1,j,:2] = copy.deepcopy(pos[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 2:4] = copy.deepcopy(vel[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 4] = copy.deepcopy(heading[:pad_len])

        # d = 2
        # heading = pred_agent[1:,:,4]
        # for k in range(heading.shape[0]):
        #     start = max(k-d,0)
        #     end = min(k+d,heading.shape[0])
        #     window = heading[start:end]
        #     heading[k] = window.sum(0)/window.shape[0]

        transformed = {}
        transformed['agent'],transformed['lane'] = self.transform_to_agent(pred_agent[0,0],pred_agent,data['lane'])
        transformed['traf'] = data['other']['traf'][0]
        transformed['pred'] = transformed['agent']
        transformed['agent'] = transformed['agent'][0]
        output = self.process_case_to_input(transformed,agent_range=300,edge_num=1000,center_num=6000)
        output['pred'] = transformed['pred']
        output['lane'] = data['lane']
        return output

    def inference_control(self, data, length = 190, per_time = 30):

        # for every x time step, pred then update

        future_traj = data['other']['agent_traj']


        pred_agent = np.zeros([length,*data['all_agent'].shape])
        pred_agent[0] = copy.deepcopy(data['all_agent'])
        # vel = (pred_agent[0,:,2]**2 + pred_agent[0,:,3]**2)**0.5
        # pred_agent[0][vel<0.05,2:4]=0
        pred_agent[1:,:,5:] = pred_agent[0,:,5:]

        #ind = list(range(0,190,5))
        pred_agent[:,0] = future_traj[0]
        agent_num = data['all_agent'].shape[0]

        for i in range(0,length-1,per_time):

            current_agent = copy.deepcopy(pred_agent[i])
            case_list = []
            for j in range(agent_num):
                a_case = {}
                a_case['agent'],a_case['lane'] = self.transform_to_agent(current_agent[j],current_agent,data['lane'])
                a_case['traf'] = data['other']['traf'][i]
                case_list.append(a_case)

            inp_list = []
            for case in case_list:
                inp_list.append(self.process_case_to_input(case))

            keys = inp_list[0].keys()

            batch = {}
            for key in keys:
                one_item = [item[key] for item in inp_list]
                batch[key] = Tensor(np.stack(one_item))

            # for inp in inp_list:
            #     draw(inp['center'], inp['agent'][inp['agent_mask'].astype(bool)][np.newaxis], context=inp['agent'][inp['agent_mask'].astype(bool)])
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    batch[key] = batch[key].cuda()
            if self.cfg['device'] == 'cuda':
                self.model.cuda()
            pred, prob = self.model(batch,False)


            best_pred_idx = torch.argmax(prob,dim=-1)
            best_pred_idx = best_pred_idx.view(pred.shape[0],1,1,1).repeat(1,1,*pred.shape[2:])
            best_pred = torch.gather(pred,dim=1,index=best_pred_idx).squeeze(1).cpu().numpy()

            # best_pred = np.concatenate([np.zeros([best_pred.shape[0],1, 2]), best_pred], axis=1)
            # best_pred = best_pred[:,1:] - best_pred[:,:-1]
            # buf = np.zeros([best_pred.shape[0],8,2])
            # cnt = 0
            # for z in range(0,40,5):
            #     slice = best_pred[:,z:z+5]
            #     buf[:,cnt] = slice.sum(-2)
            #     cnt+=1
            ## update all the agent
            for j in range(1,agent_num):
                pred_j = best_pred[j]
                agent_j = copy.deepcopy(current_agent[j])
                center = copy.deepcopy(agent_j[:2])
                center_yaw = copy.deepcopy(agent_j[4])
                rotate_theta = -(center_yaw-np.pi/2)

                # pred_j = np.concatenate([np.zeros([1,2]),pred_j],axis=0)
                # pred_j = pred_j[1:]-pred_j[:-1]
                pos = self.rotate(pred_j[:, 0], pred_j[:, 1], -rotate_theta)
                heading = -pred_j[...,-1]+center_yaw
                #heading = np.arctan2(pred_j[..., 1], pred_j[..., 0])
                #jerk = np.zeros_like(heading).astype(bool)
                #heading = np.concatenate([np.array([center_yaw]),heading])

                #vel = (pred_j[:, 0] ** 2 + pred_j[:, 1] ** 2) ** 0.5
                pos_ = np.concatenate([np.zeros([1,2]),pos],axis=0)
                vel = (pos_[1:]-pos_[:-1])/0.1
                #speed = pred_j[...,-2]*10
                speed = (vel[...,0] ** 2 + vel[...,1] ** 2) ** 0.5
                vx = (speed*np.cos(heading))[:,np.newaxis]
                vy = (speed*np.sin(heading))[:,np.newaxis]
                vel = np.concatenate([vx,vy],axis=-1)


                pos = pos + center
                pad_len = pred_agent[i+1:i+per_time+1].shape[0]
                pred_agent[i+1:i+per_time+1,j,:2] = copy.deepcopy(pos[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 2:4] = copy.deepcopy(vel[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 4] = copy.deepcopy(heading[:pad_len])

        # d = 2
        # heading = pred_agent[1:,:,4]
        # for k in range(heading.shape[0]):
        #     start = max(k-d,0)
        #     end = min(k+d,heading.shape[0])
        #     window = heading[start:end]
        #     heading[k] = window.sum(0)/window.shape[0]

        transformed = {}
        transformed['agent'],transformed['lane'] = self.transform_to_agent(pred_agent[0,0],pred_agent,data['lane'])
        transformed['traf'] = data['other']['traf'][0]
        transformed['pred'] = transformed['agent']
        transformed['agent'] = transformed['agent'][0]
        output = self.process_case_to_input(transformed,agent_range=300,edge_num=1000,center_num=6000)
        output['pred'] = transformed['pred']
        output['lane'] = data['lane']
        return output

    @time_me
    def train_one_epoch(self, train_data):
        self.current_epoch += 1
        self.model.train()
        sgd_count = 0

        for data in train_data:
            # Checking data preprocess
            for key in data.keys():
                if isinstance(data[key], torch.DoubleTensor):
                    data[key] = data[key].float()
                if isinstance(data[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    data[key] = data[key].cuda()
            self.optimizer.zero_grad()

            pred,_,loss,losses = self.model(data)


            loss_info = {}
            for k,v in losses.items():
                loss_info[k] = self.gather_loss_stat(v)
            loss_info['loss'] = self.gather_loss_stat(loss)

            loss.backward()
            self.optimizer.step()
            # log data
            if self.main_process:
                # log data in batch level
                if sgd_count % self.wandb_log_freq == 0:
                    self.print_log(loss_info)
                    self.print("\n")
                    wandb.log(loss_info)
                # * Display the results.
            sgd_count += 1
            self.total_sgd_count += 1

    def save_model(self):
        if self.current_epoch % self.save_freq == 0 and self.main_process:
            model_save_name = os.path.join(self.exp_data_path, 'saved_models',
                                           'model_{}.pt'.format(self.current_epoch))
            state = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.current_epoch
            }
            torch.save(state, model_save_name)
            self.print('\n model saved to %s' % model_save_name)

    def load_model(self, model_path,device):
        state = torch.load(model_path,map_location=torch.device(device))
        self.model.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_epoch = state["epoch"]

    @staticmethod
    def reduce_mean_on_all_proc(tensor):
        nprocs = torch.distributed.get_world_size()
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def print(self, *info):
        "Use this print instead of naive print, since we run in parallel"
        if self.main_process:
            print(*info)

    def print_log(self, log: dict):
        self.print("========== Epoch: {} ==========".format(self.current_epoch))
        for key, value in log.items():
            self.print(key, ": ", value)

    @property
    def main_process(self):
        return True if self.args.rank == 0 else False

    @property
    def distributed(self):
        return True if self.args.distributed else False

    def gather_loss_stat(self, loss):
        return loss.item() if not self.distributed else self.reduce_mean_on_all_proc(loss).item()

    def load_data(self):
        return self.train_dataloader

    def make_dir(self):
        os.makedirs(self.exp_data_path, exist_ok=False)
        os.mkdir(os.path.join(self.exp_data_path, "saved_models"))

