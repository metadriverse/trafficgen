import copy
import numpy as np
import pickle

from tqdm import tqdm
from random import choices,seed
import time
import torch
from torch import Tensor
import torch.distributed as dist
import wandb
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from utils.utils import get_time_str,time_me,transform_to_agent,from_list_to_batch,rotate,get_polygon,get_agent_pos_from_vec
from utils.typedef import AgentType

from TrafficGen_init.models.init_model import initializer
from TrafficGen_init.data_process.init_dataset import initDataset,WaymoAgent

from utils.visual_init import draw,draw_heatmap,get_heatmap,draw_metrics
from utils.visual_act import draw_seq

from TrafficGen_act.models.act_model import actuator
from TrafficGen_act.data_process.act_dataset import actDataset,process_case_to_input,process_map

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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

        self.model_type = cfg['model']

        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", rank=args.local_rank)
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
            print('[TORCH] Training in distributed mode. Process %d, local %d, total %d.' % (
                args.rank, args.local_rank, args.world_size))

        if self.model_type == 'init':
            train_dataset = initDataset(self.cfg, args)
            model = initializer(cfg)
        elif self.model_type == 'act':
            train_dataset = actDataset(self.cfg, args)
            model = actuator(cfg)
        else:
            raise NotImplementedError('no such model!')
        if len(train_dataset)>0:
            data_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                     shuffle=True,
                                     num_workers=self.cfg['num_workers'])
            self.train_dataloader = data_loader
        if args.distributed:
            model.cuda(args.local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model,
                                            device_ids=[args.local_rank],
                                            output_device=args.local_rank,
                                            broadcast_buffers=False)  # only single machine
        else:
            model = torch.nn.DataParallel(model, list(range(1)))
            model = model.to(self.cfg['device'])

        optimizer = optim.AdamW(model.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999), eps=1e-09,
                                weight_decay=self.cfg['weight_decay'], amsgrad=True)

        self.eval_data_loader = None

        if self.main_process and self.cfg["need_eval"]:
            test_set = initDataset(self.cfg, args, eval=True) if self.model_type=='init' else actDataset(self.cfg, args, eval=True)
            self.eval_data_loader = DataLoader(test_set, shuffle=False, batch_size=cfg['eval_batch_size'],
                                               num_workers=self.cfg['num_workers'])

        self.model = model
        self.in_debug = cfg["debug"]  # if in debug, wandb will log to TEST instead of cvpr
        self.optimizer = optimizer

        self.batch_size = cfg['batch_size']
        self.max_epoch = cfg['max_epoch']
        self.current_epoch = 0
        self.save_freq = save_freq
        self.exp_name = "v2_{}_{}".format(exp_name, get_time_str()) if exp_name is not None else "v2_{}".format(
            get_time_str())
        self.exp_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exps",
                                          self.exp_name)
        self.wandb_log_freq = wandb_log_freq
        self.total_sgd_count = 1
        self.training_start_time = time.time()
        #self.mmd_loss = MMD_loss()
        if self.main_process:
            self.make_dir()
            if not self.in_debug:
                wandb.login(key="a6ae178e5596edd2aa7e54d4d34abebe5759406e")
                wandb.init(
                    entity="drivingforce",
                    project="cvpr" if not self.in_debug else "TEST",
                    name=self.exp_name,
                    config=cfg)
            gpu_num = torch.cuda.device_count()
            print("gpu number:{}".format(gpu_num))
            print("gpu available:", torch.cuda.is_available())

    def train(self):
        while self.current_epoch < self.max_epoch:
            epoch_start_time = time.time()
            current_sgd_count = self.total_sgd_count
            train_data = self.train_dataloader
            _, epoch_training_time = self.train_one_epoch(train_data)
            # log data in epoch level
            if self.main_process:
                if not self.in_debug:
                    wandb.log({
                        "epoch_training_time (s)": time.time() - epoch_start_time,
                        "sgd_iters_in_this_epoch": self.total_sgd_count - current_sgd_count,
                        "epoch": self.current_epoch,
                        "epoch_training_time": epoch_training_time,
                    })

                if self.current_epoch % self.cfg["eval_frequency"] == 0:
                    self.save_model()
                    if self.cfg["need_eval"]:
                        self.eval_init()


    def get_heatmap(self,inp):

        gt_num = 4

        inp['agent_mask'][:,gt_num:] = 0
        pred, _, _ = self.model(inp,False)
        gt_idx = inp['vec_index'][:,:gt_num]
        plt = draw_heatmap(inp['center'][0],pred[0,:,0],gt_idx)

        return plt

    def process_case_for_eval(self,data_path,data_num):
        self.data_for_gen = {}
        cnt = 0
        while cnt < data_num:
            data_file_path = os.path.join(data_path, f'{cnt}.pkl')

            with open(data_file_path, 'rb+') as f:
                datas = pickle.load(f)

            self.data_for_gen[cnt] = self.process(datas)

            cnt+=1

    def transform_coordinate_map(self,data,lane):
        """
        Every frame is different
        """
        pos = data['sdc_pos'][0]
        lane[..., :2] -= pos
        sdc_theta = data['sdc_theta'][0]
        x = lane[..., 0]
        y = lane[..., 1]
        x_transform = np.cos(sdc_theta) * x - np.sin(sdc_theta) * y
        y_transform = np.cos(sdc_theta) * y + np.sin(sdc_theta) * x
        output_coords = np.stack((x_transform, y_transform), axis=-1)
        lane[..., :2] = output_coords

        return lane

    def process(self,data):
        case_info = {}

        data['lane'] = self.transform_coordinate_map(data,data['lane'])
        data['unsampled_lane'] = self.transform_coordinate_map(data, data['unsampled_lane'])
        # case_info

        scene = WaymoScene(data)
        # agent: x,y,vx,vy,l,w,sin_head,cos_head
        case_info["agent"] = scene.agent_data
        case_info['agent'][..., 4] = 5.286
        case_info['agent'][..., 5] = 2.332
        case_info["agent_mask"] = scene.agent_mask

        case_info['center'], case_info['center_mask'], case_info['bound'], case_info['bound_mask'], \
        case_info['cross'], case_info['cross_mask'], case_info['rest'] = self.process_map(data)

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
        for i in range(b):
            mask = agent_mask[i]
            agent_info = padding_content[i]
            indx = vec_ind[i]
            indx = indx[mask.to(bool)]
            gt[i, indx, 0] = 1
            gt[i, indx, 1:] = agent_info[mask.to(bool)]

        case_info['gt'] = gt
        case_info['line_with_agent'] = line_with_agent

        ind = list(range(0, 190, 10))
        lane = data['lane'][ind]
        case_info['lane'] = lane
        case_list = []

        for i in range(1):
            dic = {}
            for k, v in case_info.items():
                dic[k] = v[i]
            case_list.append(dic)

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

        case_list[0]['traf'] = data['traf_p_c_f']
        case_list[0]['center_info'] = data['center_info']
        case_list[0]['unsampled_lane'] = lane
        return case_list

    def draw_generation_process(self,vis=True, save=False):
        save_path = '/Users/fenglan/Downloads/waymo/onemap'
        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader
            cnt = 0

            for batch in eval_data:
                seed(cnt)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()

                output, heat_maps = self.inference(batch)

                if vis:
                    if not os.path.exists('./vis/heatmap'):
                        os.mkdir('./vis/heatmap')
                    path = f'./vis/heatmap/{cnt}'
                    if not os.path.exists(path):
                        os.mkdir(path)

                    center = batch['center'][0].cpu().numpy()
                    rest = batch['rest'][0].cpu().numpy()
                    bound = batch['bound'][0].cpu().numpy()
                    pred_agent = output['agent']
                    for j in range(len(pred_agent)-1):
                        output_path = os.path.join(path,f'{j}')
                        draw(center, heat_maps[j],pred_agent,rest, edge=bound,save=True,path=output_path)

                # generate case
                if save:
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].cpu().numpy()
                    output = {}
                    case_agent = np.zeros([pred_agent.shape[0],9])
                    case_agent[:,-2:]=1
                    case_agent[:,:4] = pred_agent[:,:4]
                    case_agent[:,5:7] = pred_agent[:,4:6]
                    case_agent[:,4] = -np.arctan2(pred_agent[:,6],pred_agent[:,7])+np.pi/2
                    #output['lane'] = batch['lane'][0]
                    output['all_agent'] = case_agent
                    output['other'] = batch['other']
                    output['lane'] = batch['other']['lane']
                    output['other'].pop('lane')
                    output['gt_agent'] = inp['agent'][0].numpy()

                    p = os.path.join(save_path, f'{i}.pkl')
                    with open(p, 'wb') as f:
                        pickle.dump(output, f)

                cnt+=1

    def process_train_to_eval(self,data):


        sdc_theta = data['sdc_theta']
        pos = data['sdc_pos']

        agent_type = data['nbrs_p_c_f'][:,0,-2]==1
        pred_mask = data['pred_list'].astype(bool)
        agent_mask = agent_type*pred_mask
        other = data['nbrs_p_c_f'][agent_mask]

        all_agent = np.concatenate(
            [data['ego_p_c_f'][np.newaxis], other], axis=0)
        coord = rotate(all_agent[..., 0], all_agent[..., 1], -sdc_theta) + pos
        vel = rotate(all_agent[..., 2], all_agent[..., 3], -sdc_theta)

        yaw = -sdc_theta+np.pi/2

        all_agent[..., 4] = all_agent[..., 4] + yaw
        all_agent[..., :2] = coord
        all_agent[..., 2:4] = vel

        #for j in range(all_agent.shape[0]):
        agents = copy.deepcopy(all_agent)

        pos0 = copy.deepcopy(agents[0, 0, :2])
        theta0 = -(copy.deepcopy(agents[0,0, 4]))

        coord = agents[..., :2]
        vel = agents[..., 2:4]
        coord -= pos0

        coord = rotate(coord[..., 0], coord[..., 1], theta0)
        vel = rotate(vel[..., 0], vel[..., 1], theta0)

        agents[..., :2] = coord
        agents[..., 2:4] = vel
        agents[..., 4] = agents[..., 4] - copy.deepcopy(agents[0][0, 4])

        # then recover lane's position
        lane = copy.deepcopy(data['lane'])
        lane[..., :2] -= pos0

        output_coords = rotate(lane[..., 0], lane[..., 1], theta0)

        lane[..., :2] = output_coords

        output = {}
        output['lane'] = lane
        output['all_agent'] = agents[:,0]
        other = {}
        # one_case['traf'] = data['traf_p_c_f'][i]

        other['traf'] = data['traf_p_c_f']
        output['other'] = other

        return output

    def eval_act(self):
        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader.dataset
            cnt = 0
            eval_results = []
            for i in tqdm(range(len(eval_data))):


                batch = copy.deepcopy(eval_data[i])
                inp = self.process_train_to_eval(batch)

                for key in batch.keys():
                    if isinstance(batch[key], np.ndarray):
                        batch[key] = Tensor(batch[key])
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                cnt += 1
                #inp = copy.deepcopy(batch)

                output = self.inference_control(inp,ego_gt=False,length=90)

                lane = inp['lane']
                traf = inp['other']['traf']
                agent = output['pred']
                dir_path = f'./vis/gif/{i}'
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                ind = list(range(0,90,5))
                #agent = np.delete(agent,4,axis=1)
                #del heat_map[5]
                agent = agent[ind]
                for t in range(agent.shape[0]):
                    agent_t = agent[t]
                    path = os.path.join(dir_path, f'{t}')
                    traf_t = traf[t]

                    cent,cent_mask,bound,bound_mask,_,_,rest = process_map(lane,traf_t, 2000, 1000, 70, 0)

                    draw_seq(cent,bound,rest,agent_t,path=path,save=True)

                loss = self.metrics(output,inp)
                eval_results.append(loss)
        return

    def eval_init(self):
        self.model.eval()
        eval_results = []
        with torch.no_grad():
            eval_data = self.eval_data_loader
            cnt = 0

            imgs = {}
            for batch in eval_data:
                if cnt>10:break
                seed(cnt)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                output, heat_maps = self.inference(batch)

                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()
                pred_agent = output['agent']

                imgs[f'vis_{cnt}'] = wandb.Image(draw(center, pred_agent, rest, edge=bound))
                cnt+=1

            log = {}
            log['imgs'] = imgs
            if not self.in_debug:
                wandb.log(log)
            #     if len(output['prob'])<=1:continue
            #     loss = self.metrics(output,batch)
            #     eval_results.append(loss)
            #
            # loss = torch.zeros([4,10])
            # for i in range(10):
            #     prob_i=0
            #     vel_i=0
            #     dir_i=0
            #     coord_i=0
            #     cnt=0
            #     for result in eval_results:
            #         if len(result['prob'])<(i+1): continue
            #         prob_i+=result['prob'][i]
            #         vel_i += result['vel'][i]
            #         coord_i += result['coord'][i]
            #         dir_i += result['dir'][i]
            #         cnt+=1
            #     loss[0,i] = prob_i/cnt
            #     loss[1,i] = coord_i/cnt
            #     loss[2,i] = vel_i/cnt
            #     loss[3,i] = dir_i/cnt
            #
            # loss = loss.numpy()
            # p = f'./{self.exp_name}'
            # with open(p, 'wb') as f:
            #     pickle.dump(loss, f)
            # plt_pred = wandb.Image(draw_metrics(loss))

    def metrics(self,pred,gt):
        #gt.pop('other')
        for k,v in gt.items():
            gt[k] = gt[k].squeeze(0).cpu()

        gt_agent = gt['agent_feat'][1:]
        agent_num = gt['agent_mask'].shape[0]
        line_mask = gt['center_mask'].to(bool)
        vec_indx = gt['vec_index'].to(int)
        gt_prob = gt['gt'][:,0][line_mask]
        pred_prob = torch.Tensor(np.stack(pred['prob']))[:agent_num-1]
        pred_prob = pred_prob[:,line_mask]
        pred_agent = pred['agent']

        BCE = torch.nn.BCEWithLogitsLoss()
        MSE = torch.nn.MSELoss()
        L1 = torch.nn.L1Loss()

        gt_prob[vec_indx[0]]=0

        bce_list = []
        coord_list = []
        vel_list = []
        dir_list = []
        for i in range(agent_num-1):

            bce_loss = BCE(pred_prob[i],gt_prob)
            gt_prob[vec_indx[i+1]]=0

            pred_coord = pred_agent[i]['coord']
            pred_vel = pred_agent[i]['vel']
            pred_dir = pred_agent[i]['agent_dir']

            gt_coord = gt_agent[i,:2]
            gt_vel = gt_agent[i,2:4]
            gt_dir = gt_agent[i,6:8]

            coord_loss = MSE(gt_coord,pred_coord)
            vel_loss = MSE(gt_vel,pred_vel)
            dir_loss = L1(gt_dir,pred_dir)

            bce_list.append(bce_loss)
            coord_list.append(coord_loss)
            vel_list.append(vel_loss)
            dir_list.append(dir_loss)

        metrics = {}
        metrics['prob'] = bce_list
        metrics['vel'] = vel_list
        metrics['coord'] = coord_list
        metrics['dir'] = dir_list
        return metrics

    def sample_from_distribution(self, pred,center_lane,repeat_num=10):
        prob = pred['prob'][0]

        pos = pred['pos'].sample()
        pos_logprob = pred['pos'].log_prob(pos)

        heading = pred['heading'].sample()
        heading_logprob = pred['heading'].log_prob(heading)

        vel_heading = pred['vel_heading'].sample()
        vel_heading_logprob = pred['vel_heading'].log_prob(vel_heading)

        bbox = pred['bbox'].sample()
        bbox_logprob = pred['bbox'].log_prob(bbox)

        speed = pred['speed'].sample()
        speed_logprob = pred['speed'].log_prob(speed)

        agents = get_agent_pos_from_vec(center_lane, pos[0], speed[0], vel_heading[0], heading[0], bbox[0])

        idx_list = []
        prob_list = []
        for i in range(repeat_num):
            indx = choices(list(range(prob.shape[-1])), prob)[0]
            vec_logprob_ = prob[indx]
            pos_logprob_ = pos_logprob[0,indx]
            heading_logprob_ = heading_logprob[0,indx]
            vel_heading_logprob_ = vel_heading_logprob[0,indx]
            bbox_logprob_ = bbox_logprob[0,indx]
            speed_logprob_ = speed_logprob[0,indx]
            all_prob = vec_logprob_+pos_logprob_+heading_logprob_+vel_heading_logprob_+heading_logprob_+bbox_logprob_+speed_logprob_
            prob_list.append(all_prob)
            idx_list.append(indx)

        max_indx = np.argmax(prob_list)
        the_indx = idx_list[max_indx]

        return agents,prob,the_indx

    def inference(self, data, eval=False):

        agent_num = data['agent_mask'].sum().item()

        idx_list = []

        vec_indx = data['vec_based_rep'][...,0]
        idx_list.append(vec_indx[0,0].item())
        shapes = []

        ego_agent = data['agent'][0,[0]].numpy()
        ego_agent = WaymoAgent(ego_agent)
        ego_poly = ego_agent.get_polygon()[0]
            #get_polygon(ego_agent[:2],ego_agent[4],ego_agent[5] + 0.1, ego_agent[6] + 0.1)
        shapes.append(ego_poly)

        virtual_list = []
        minimum_agent = self.cfg['pad_num']
        center = data['center'][0]
        center_mask = data['center_mask'][0]
        pred_list = []
        pred_list.append(ego_agent)
        heat_maps = []
        prob_list = []

        for i in range(1,max(agent_num,minimum_agent)):
            data['agent_mask'][:, :i] = 1
            data['agent_mask'][:,i:]=0

            pred, _, _ = self.model(data, False)
            pred['prob'][:,idx_list]=0

            while True:
                agents,prob, indx = self.sample_from_distribution(pred,center)
                the_agent = agents.get_agent(indx)

                poly = the_agent.get_polygon()[0]

                intersect = False
                for shape in shapes:
                    if poly.intersects(shape):
                        intersect = True
                        break
                if not intersect:
                    shapes.append(poly)
                    break
                else: continue

            pred_list.append(the_agent)
            data['agent_feat'][:,i] = Tensor(the_agent.get_inp())
            idx_list.append(indx)


            heat_maps.append(get_heatmap(agents.position[:,0][center_mask], agents.position[:, 1][center_mask],prob[center_mask], 20))
            prob_list.append(prob)

        output = {}
        output['agent'] = pred_list
        output['prob'] = prob_list

        return output, heat_maps
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

            pred, loss,losses = self.model(data)

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
                    if not self.in_debug:
                        wandb.log(loss_info)
                # * Display the results.
            sgd_count += 1
            self.total_sgd_count += 1

    def draw_gifs(self,vis=True,save_path=None):
        self.model.eval()
        with torch.no_grad():
            eval_data = self.eval_data_loader

            scene_data = eval_data.dataset.scene_data
            pred_list = []

            for i in tqdm(range(len(scene_data))):
                data_inp = scene_data[i]
                #heat_map = data_inp['heat_map']
                pred_i = self.inference_control(data_inp)
                #pred_i['pred']=np.delete(pred_i['pred'],4,axis=1)
                pred_list.append(pred_i)

                if vis:
                    other = scene_data[i]['other']
                    lane = pred_i['lane']
                    traf = other['traf']
                    agent = pred_i['pred']
                    dir_path = f'./vis/gif/{i}'
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)
                    ind = list(range(0,190,5))
                    #agent = np.delete(agent,4,axis=1)
                    #del heat_map[5]
                    agent = agent[ind]
                    for t in range(agent.shape[0]):
                    #for t in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]:
                        agent_t = agent[t]
                        #path = os.path.join(dir_path,f'{t+agent_t.shape[0]-1}')
                        path = os.path.join(dir_path, f'{t}')
                        traf_t = traf[t]
                        inp = {}
                        inp['traf'] = traf_t
                        inp['lane'] = lane
                        cent,cent_mask,bound,bound_mask,_,_,rest = process_map(inp, 2000, 1000, 70, 0)
                        #cent,cent_mask,bound,bound_mask,_,_,rest = WaymoDataset.process_map(inp,2000,1000,100)
                        # if t==0:
                        #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                        #     for k in range(1,agent_t.shape[0]):
                        #         heat_path = os.path.join(dir_path, f'{k-1}')
                        #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)
                        draw_seq(cent,bound,rest,agent_t,path=path,save=True)

            if save_path:
                self.save_as_metadrive_data(pred_list,scene_data,save_path)

    def inference_control(self, data, ego_gt=True,length = 190, per_time = 10):
        # for every x time step, pred then update

        pred_agent = np.zeros([length,*data['all_agent'].shape])

        pred_agent[0] = copy.deepcopy(data['all_agent'])
        pred_agent[1:,:,5:] = pred_agent[0,:,5:]
        agent_num = data['all_agent'].shape[0]
        start_idx = 0

        if ego_gt==True:
            future_traj = data['other']['agent_traj']
            pred_agent[:,0] = future_traj[0]
            start_idx = 1

        for i in range(0,length-1,per_time):

            current_agent = copy.deepcopy(pred_agent[i])
            case_list = []
            for j in range(agent_num):
                a_case = {}
                a_case['agent'],a_case['lane'] = transform_to_agent(current_agent[j],current_agent,data['lane'])
                a_case['traf'] = data['other']['traf'][i]
                case_list.append(a_case)

            inp_list = []
            for case in case_list:
                inp_list.append(process_case_to_input(case))

            batch = from_list_to_batch(inp_list)
            # for inp in inp_list:
            #     draw_seq(center=inp['center'],context=inp['agent'][inp['agent_mask'].astype(bool)],edge=inp['bound'],rest=inp['rest'],gt=True)
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    batch[key] = batch[key].cuda()
            if self.cfg['device'] == 'cuda':
                self.model.cuda()

            pred = self.model(batch,False)
            prob = pred['prob']
            velo_pred = pred['velo']
            pos_pred = pred['pos']
            heading_pred = pred['heading']
            all_pred = torch.cat([pos_pred,velo_pred,heading_pred.unsqueeze(-1)],dim=-1)

            best_pred_idx = torch.argmax(prob,dim=-1)
            best_pred_idx = best_pred_idx.view(agent_num,1,1,1).repeat(1,1,*all_pred.shape[2:])
            best_pred = torch.gather(all_pred,dim=1,index=best_pred_idx).squeeze(1).cpu().numpy()

            ## update all the agent
            for j in range(start_idx,agent_num):
                pred_j = best_pred[j]
                agent_j = copy.deepcopy(current_agent[j])
                center = copy.deepcopy(agent_j[:2])
                center_yaw = copy.deepcopy(agent_j[4])
                rotate_theta = -(center_yaw-np.pi/2)

                pos = rotate(pred_j[:, 0], pred_j[:, 1], -rotate_theta)
                heading = pred_j[...,-1]+center_yaw
                vel = rotate(pred_j[:, 2], pred_j[:, 3], -rotate_theta)

                pos = pos + center
                pad_len = pred_agent[i+1:i+per_time+1].shape[0]
                pred_agent[i+1:i+per_time+1,j,:2] = copy.deepcopy(pos[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 2:4] = copy.deepcopy(vel[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 4] = copy.deepcopy(heading[:pad_len])

        transformed = {}
        transformed['agent'],transformed['lane'] = transform_to_agent(pred_agent[0,0],pred_agent,data['lane'])
        transformed['traf'] = data['other']['traf'][0]
        transformed['pred'] = transformed['agent']
        transformed['agent'] = transformed['agent'][0]
        output = process_case_to_input(transformed,agent_range=300,edge_num=1000,center_num=6000)
        output['pred'] = transformed['pred']
        output['lane'] = data['lane']
        return output

    def save_model(self):
        if self.current_epoch % self.save_freq == 0 and self.main_process:
            if self.model_type=='act':
                model_name = f'act_{self.current_epoch}'
            else:
                model_name = f'init_{self.current_epoch}'

            model_save_name = os.path.join(self.exp_data_path, 'saved_models', model_name)
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

    def make_dir(self):
        os.makedirs(self.exp_data_path, exist_ok=False)
        os.mkdir(os.path.join(self.exp_data_path, "saved_models"))
