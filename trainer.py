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

from utils.utils import get_time_str,time_me,transform_to_agent,from_list_to_batch,rotate,normalize_angle
from utils.typedef import AgentType
from metrics.mmd.mmd import MMD

from TrafficGen_init.models.init_model import initializer
from TrafficGen_init.models.sceneGen import sceneGen
from TrafficGen_init.data_process.init_dataset import initDataset,WaymoAgent

from utils.visual_init import draw,draw_seq

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
        elif self.model_type == 'sceneGen':
            train_dataset = initDataset(self.cfg, args)
            model = sceneGen(cfg)
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
            test_set = actDataset(self.cfg, args, eval=True) if self.model_type=='act' else initDataset(self.cfg, args, eval=True)
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

    def get_metrics(self):
        self.model.eval()
        device = self.cfg['device']
        eval_data = self.eval_data_loader
        with torch.no_grad():
            mmd_metrics = {'heading': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                           'size': MMD(device=device,kernel_mul=1.0, kernel_num=1),
                           'speed': MMD(device=device,kernel_mul=1.0, kernel_num=1),
                           'position': MMD(device=device,kernel_mul=1.0, kernel_num=1)}
            cnt = 0
            for batch in tqdm(eval_data):

                seed(cnt)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                target_agent = copy.deepcopy(batch['agent'])

                output= self.model(batch,eval=True)
                pred_agent = output['agent']
                agent_num = len(pred_agent)
                pred_agent = pred_agent[1:]
                device = batch['center'].device
                source = {
                    'heading': torch.tensor(normalize_angle(np.concatenate([x.heading for x in pred_agent], axis=0)),
                                            device=device),
                    'size': torch.tensor(np.concatenate([x.length_width for x in pred_agent], axis=0), device=device),
                    'speed': torch.tensor(np.concatenate([x.velocity for x in pred_agent], axis=0), device=device),
                    'position': torch.tensor(np.concatenate([x.position for x in pred_agent], axis=0), device=device)}
                if torch.any(torch.isnan(source['speed'])):
                    print('nan!')
                    continue
                target = {'heading': normalize_angle(target_agent[0, 1:agent_num, [4]]),
                          'size': target_agent[0, 1:agent_num, 5:7],
                          'speed': target_agent[0, 1:agent_num, 2:4],
                          'position': target_agent[0, 1:agent_num, :2]}

                for attr, metri in mmd_metrics.items():
                    # ignore empty scenes
                    if agent_num <= 1:
                        continue
                    if torch.all(source[attr]==0) and torch.all(target[attr]==0):
                        continue
                    metri.update(source[attr], target[attr])

            log = {}
            for attr, metric in mmd_metrics.items():
                log[attr] = metric.compute()
            print(log)
            if not self.in_debug:
                wandb.log(log)

    def get_cases(self):
        context_num = 1

        if not os.path.exists('./cases/initialized'):
            os.mkdir('./cases/initialized')
        save_path = './cases/initialized'
        self.model.eval()
        eval_data = self.eval_data_loader
        with torch.no_grad():
            cnt = 0
            for data in tqdm(eval_data):
                #seed(cnt)
                # if cnt>70:
                #     break
                #     cnt+=1
                #     continue
                #for r in range(20):
                batch = copy.deepcopy(data)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()

                output= self.model(batch,eval=True,context_num=context_num)
                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()
                output_path = os.path.join('./cases/vis',f'{cnt}')
                draw(center, output['agent'], other=rest, edge=bound, save=True,
                     path=output_path)

                pred_agent = output['agent']
                agent = np.concatenate([x.get_inp(act=True) for x in pred_agent], axis=0)
                agent = agent[:32]
                agent_num = agent.shape[0]
                agent = np.pad(agent,([0,32-agent_num],[0,0]))
                agent_mask = np.zeros([agent_num])
                agent_mask = np.pad(agent_mask, ([0, 32 - agent_num]))
                agent_mask[:agent_num]=1
                agent_mask = agent_mask.astype(bool)

                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cpu().numpy()
                output = {}
                output['context_num'] = context_num
                output['all_agent'] = agent
                output['agent_mask'] = agent_mask
                output['lane'] = batch['other']['lane'][0].cpu().numpy()
                output['unsampled_lane'] = batch['other']['unsampled_lane'][0].cpu().numpy()
                output['traf'] = self.eval_data_loader.dataset[0]['other']['traf']
                output['gt_agent'] = batch['other']['gt_agent'][0].cpu().numpy()
                output['gt_agent_mask'] = batch['other']['gt_agent_mask'][0].cpu().numpy()

                p = os.path.join(save_path, f'{cnt}.pkl')
                with open(p, 'wb') as f:
                    pickle.dump(output, f)
                cnt+=1
        return

    def get_heatmaps(self):
        if not os.path.exists('./vis/heatmap'):
            os.mkdir('./vis/heatmap')
        self.model.eval()
        eval_data = self.eval_data_loader
        with torch.no_grad():
            cnt = 0
            for batch in tqdm(eval_data):
                seed(cnt)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                output= self.model(batch,eval=True)
                heat_maps = output['heat_maps']
                pred_agent = output['agent']
                path = f'./vis/heatmap/{cnt}'
                if not os.path.exists(path):
                    os.mkdir(path)

                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()

                for j in range(len(pred_agent) - 1):
                    output_path = os.path.join(path, f'{j}')
                    draw(center,  pred_agent[:j+1], other=rest, heat_map=heat_maps[j],edge=bound, save=True, path=output_path)
                cnt+=1
        return

    def eval_init(self):
        self.model.eval()
        eval_data = self.eval_data_loader
        with torch.no_grad():
            cnt = 0
            imgs = {}
            for batch in eval_data:
                seed(cnt)
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                output= self.model(batch,eval=True)

                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()
                pred_agent = output['agent']
                if cnt < 30:
                    imgs[f'vis_{cnt}'] = wandb.Image(draw(center, pred_agent, rest, edge=bound))

                cnt += 1
            log = {}
            log['imgs'] = imgs
            if not self.in_debug:
                wandb.log(log)

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
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

    def get_gifs(self,vis=True, snapshot=True):
        self.model.eval()

        with torch.no_grad():
            pred_list = []
            #for i in tqdm([2,8,9,20,58,99,101,109,131,156,159,169,170]):
            for i in tqdm([17]):
                with open(f'./cases/initialized/{i}.pkl', 'rb+') as f:
                    data = pickle.load(f)

                pred_i = self.inference_control(data)

                pred_i=np.delete(pred_i,3,axis=1)

                pred_list.append(pred_i)
                if vis:

                    if snapshot:
                        dir_path = f'./vis/snapshots/{i}'
                        ind = list(range(0,190,10))
                        agent = pred_i[ind]
                        #agent = agent[:,[6,7,10,12]]

                        agent_0 = agent[0]
                        agent0_list = []
                        agent_num = agent_0.shape[0]
                        for a in range(agent_num):
                            agent0_list.append(WaymoAgent(agent_0[[a]]))

                        cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(data['lane'][np.newaxis],
                                                                                        [data['traf'][0]],
                                                                                        center_num=256, edge_num=128,
                                                                                        offest=0, lane_range=60)
                        # draw_dict = {}
                        # draw_dict['a'] = cent[0]
                        # draw_dict['b'] = agent0_list
                        # draw_dict['c'] = agent[...,:2]
                        # draw_dict['d'] = bound[0]
                        # draw_dict['e'] = rest[0]
                        #
                        # with open('./draw.pkl', 'wb') as f:
                        #     pickle.dump(draw_dict, f)
                        draw_seq(cent[0],agent0_list,agent[...,:2],edge=bound[0],other=rest[0],path=dir_path,save=True)


                    else:

                        dir_path = f'./vis/gif/{i}'
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)

                        ind = list(range(0,190,5))
                        agent = pred_i[ind]
                        for t in range(agent.shape[0]):
                            agent_t = agent[t]
                            agent_list = []
                            for a in range(agent_t.shape[0]):
                                agent_list.append(WaymoAgent(agent_t[[a]]))

                            path = os.path.join(dir_path, f'{t}')
                            cent,cent_mask,bound,bound_mask,_,_,rest,_ = process_map(data['lane'][np.newaxis],[data['traf'][int(t*5)]], center_num=256, edge_num=128,offest=0, lane_range=60)
                            draw(cent[0],agent_list,edge=bound[0],other=rest[0],path=path,save=True)


                        # if t==0:
                        #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                        #     for k in range(1,agent_t.shape[0]):
                        #         heat_path = os.path.join(dir_path, f'{k-1}')
                        #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)
            # if save_path:
            #     self.save_as_metadrive_data(pred_list,scene_data,save_path)

    def get_gifs_from_gt(self,vis=True, snapshot=True):
        self.model.eval()
        eval_data = self.eval_data_loader.dataset
        with torch.no_grad():
            cnt = 0
            for data in eval_data:
                if vis:
                    if snapshot:
                        dir_path = f'./vis/snapshots/{cnt}'
                        cnt+=1
                        ind = list(range(0,120,10))
                        agent = data['all_valid'][:120]
                        agent = agent[ind]
                        agent_0 = agent[0]
                        agent0_list = []
                        for a in range(agent_0.shape[0]):
                            agent0_list.append(WaymoAgent(agent_0[[a]]))
                        draw_seq(data['center'],agent0_list,agent[...,:2],edge=data['bound'],other=data['rest'],path=dir_path,save=True)


                    else:

                        dir_path = f'./vis/gif/{i}'
                        if not os.path.exists(dir_path):
                            os.mkdir(dir_path)

                        ind = list(range(0,190,5))
                        agent = pred_i[ind]
                        for t in range(agent.shape[0]):
                            agent_t = agent[t]
                            agent_list = []
                            for a in range(agent_t.shape[0]):
                                agent_list.append(WaymoAgent(agent_t[[a]]))

                            path = os.path.join(dir_path, f'{t}')
                            cent,cent_mask,bound,bound_mask,_,_,rest,_ = process_map(data['lane'][np.newaxis],[data['traf'][int(t*5)]], center_num=256, edge_num=128,offest=0, lane_range=60)
                            draw(cent[0],agent_list,edge=bound[0],other=rest[0],path=path,save=True)


                        # if t==0:
                        #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                        #     for k in range(1,agent_t.shape[0]):
                        #         heat_path = os.path.join(dir_path, f'{k-1}')
                        #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)
            # if save_path:
            #     self.save_as_metadrive_data(pred_list,scene_data,save_path)

    def get_metrics_for_act(self):
        self.model.eval()
        eval_data = self.eval_data_loader

        mean_ade=0
        mean_fde=0
        with torch.no_grad():
            cnt = 0
            for batch in tqdm(eval_data):
                for key in batch.keys():
                    if isinstance(batch[key], torch.DoubleTensor):
                        batch[key] = batch[key].float()
                    if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                        batch[key] = batch[key].cuda()
                pred = self.model(batch, False)
                prob = pred['prob']
                velo_pred = pred['velo']
                pos_pred = pred['pos']
                heading_pred = pred['heading']
                all_pred = torch.cat([pos_pred, velo_pred, heading_pred.unsqueeze(-1)], dim=-1)
                bs = prob.shape[0]
                best_pred_idx = torch.argmax(prob, dim=-1)
                best_pred_idx = best_pred_idx.view(bs, 1, 1, 1).repeat(1, 1, *all_pred.shape[2:])
                best_pred = torch.gather(all_pred, dim=1, index=best_pred_idx).squeeze(1)

                pred_pos = best_pred[...,:2]
                gt_pos = batch['gt_pos']

                de1 = pred_pos-gt_pos
                de1 = (de1[...,0]**2+de1[...,1]**2)**0.5
                fde = de1[:,-1].mean().item()
                ade = de1.mean().item()
                mean_ade = (mean_ade+ade)/(cnt+1)
                mean_fde = (mean_fde+fde)/(cnt+1)
                cnt+=1
            print(mean_ade)
            print(mean_fde)
        return

    def inference_control(self, data, ego_gt=True,length = 190, per_time = 20):
        # for every x time step, pred then update
        agent_num = data['agent_mask'].sum()
        data['agent_mask'] = data['agent_mask'][:agent_num]
        data['all_agent'] = data['all_agent'][:agent_num]

        pred_agent = np.zeros([length,agent_num,8])
        pred_agent[0,:,:7] = copy.deepcopy(data['all_agent'])
        pred_agent[1:,:,5:7] = pred_agent[0,:,5:7]

        start_idx = 0

        if ego_gt==True:
            future_traj = data['gt_agent']
            pred_agent[:,0,:7] = future_traj[:,0]
            start_idx = 1

        for i in range(0,length-1,per_time):

            current_agent = copy.deepcopy(pred_agent[i])
            case_list = []
            for j in range(agent_num):
                a_case = {}
                a_case['agent'],a_case['lane'] = transform_to_agent(current_agent[j],current_agent,data['lane'])
                a_case['traf'] = data['traf'][i]
                case_list.append(a_case)

            inp_list = []
            for case in case_list:
                inp_list.append(process_case_to_input(case))
            batch = from_list_to_batch(inp_list)
            # for inp in inp_list:
            #     all_ = inp['agent'][inp['agent_mask'].astype(bool)]
            #     agent_list = []
            #     for i in range(all_.shape[0]):
            #         agent_list.append(WaymoAgent(all_[[i]],from_inp=True))
            #     draw(inp['center'],  agent_list, other=inp['rest'],edge=inp['bound'], save=False)
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

                pos = rotate(pred_j[:, 0], pred_j[:, 1], center_yaw)
                heading = pred_j[...,-1]+center_yaw
                vel = rotate(pred_j[:, 2], pred_j[:, 3], center_yaw)

                pos = pos + center
                pad_len = pred_agent[i+1:i+per_time+1].shape[0]
                pred_agent[i+1:i+per_time+1,j,:2] = copy.deepcopy(pos[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 2:4] = copy.deepcopy(vel[:pad_len])
                pred_agent[i+1:i + per_time+1, j, 4] = copy.deepcopy(heading[:pad_len])

        # transformed = {}
        # #transformed['agent'],transformed['lane'] = transform_to_agent(pred_agent[0,0],pred_agent,data['lane'])
        # transformed['agent'], transformed['lane'] = pred_agent, data['lane']
        # #transformed['traf'] = data['traf'][0]
        # transformed['pred'] = transformed['agent']
        # transformed['agent'] = transformed['agent'][0]
        # output = process_case_to_input(transformed,agent_range=300,edge_num=1000,center_num=6000)
        # output['pred'] = transformed['pred']
        # output['lane'] = data['lane']
        return pred_agent

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

    # def metrics(self,pred,gt):
    #     #gt.pop('other')
    #     for k,v in gt.items():
    #         gt[k] = gt[k].squeeze(0).cpu()
    #
    #     gt_agent = gt['agent_feat'][1:]
    #     agent_num = gt['agent_mask'].shape[0]
    #     line_mask = gt['center_mask'].to(bool)
    #     vec_indx = gt['vec_index'].to(int)
    #     gt_prob = gt['gt'][:,0][line_mask]
    #     pred_prob = torch.Tensor(np.stack(pred['prob']))[:agent_num-1]
    #     pred_prob = pred_prob[:,line_mask]
    #     pred_agent = pred['agent']
    #
    #     BCE = torch.nn.BCEWithLogitsLoss()
    #     MSE = torch.nn.MSELoss()
    #     L1 = torch.nn.L1Loss()
    #
    #     gt_prob[vec_indx[0]]=0
    #
    #     bce_list = []
    #     coord_list = []
    #     vel_list = []
    #     dir_list = []
    #     for i in range(agent_num-1):
    #
    #         bce_loss = BCE(pred_prob[i],gt_prob)
    #         gt_prob[vec_indx[i+1]]=0
    #
    #         pred_coord = pred_agent[i]['coord']
    #         pred_vel = pred_agent[i]['vel']
    #         pred_dir = pred_agent[i]['agent_dir']
    #
    #         gt_coord = gt_agent[i,:2]
    #         gt_vel = gt_agent[i,2:4]
    #         gt_dir = gt_agent[i,6:8]
    #
    #         coord_loss = MSE(gt_coord,pred_coord)
    #         vel_loss = MSE(gt_vel,pred_vel)
    #         dir_loss = L1(gt_dir,pred_dir)
    #
    #         bce_list.append(bce_loss)
    #         coord_list.append(coord_loss)
    #         vel_list.append(vel_loss)
    #         dir_list.append(dir_loss)
    #
    #     metrics = {}
    #     metrics['prob'] = bce_list
    #     metrics['vel'] = vel_list
    #     metrics['coord'] = coord_list
    #     metrics['dir'] = dir_list
    #     return metrics