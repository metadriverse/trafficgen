import copy
import numpy as np
import pickle

from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trafficgen.utils.utils import transform_to_agent,from_list_to_batch,rotate
import imageio

from TrafficGen_init.models.init_distribution import initializer
from TrafficGen_init.data_process.init_dataset import initDataset,WaymoAgent

from trafficgen.utils.visual_init import draw,draw_seq

from TrafficGen_act.models.act_model import actuator
from TrafficGen_act.data_process.act_dataset import process_case_to_input,process_map

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

        model1 = initializer(cfg['init_model'])
        model2 = actuator()

        model1 = torch.nn.DataParallel(model1, list(range(1)))
        model1 = model1.to(cfg['device'])
        model2 = torch.nn.DataParallel(model2, list(range(1)))
        model2 = model2.to(cfg['device'])
        self.model1 = model1
        self.model2 = model2

        test_init_dataset = initDataset(cfg)
        self.eval_init_loader = DataLoader(test_init_dataset, shuffle=False, batch_size=1, num_workers=0)


    def load_model(self, model,model_path, device):
        state = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state["state_dict"])
    def print(self, *info):
        "Use this print instead of naive print, since we run in parallel"
        if self.main_process:
            print(*info)

    def print_log(self, log: dict):
        self.print("========== Epoch: {} ==========".format(self.current_epoch))
        for key, value in log.items():
            self.print(key, ": ", value)
    def wash(self,batch):
        for key in batch.keys():
            if isinstance(batch[key], np.ndarray):
                batch[key] = Tensor(batch[key])
            if isinstance(batch[key], torch.DoubleTensor):
                batch[key] = batch[key].float()
            if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                batch[key] = batch[key].cuda()
            if 'mask' in key:
                batch[key] = batch[key].to(bool)

    def generate_scenarios(self):
        # generate temp data in ./cases/initialized, and visualize in ./vis/initialized
        self.place_vehicles(vis=True)

        # generate trajectory from temp data, and visualize in ./vis/snapshots.
        # set gif to True to generate gif in ./vis/gif
        self.generate_traj(snapshot=True,gif=False)

    def place_vehicles(self,vis=True):
        context_num = 1

        vis_path = './vis/initialized'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        save_path = './cases/initialized'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model1.eval()
        eval_data = self.eval_init_loader
        with torch.no_grad():
            for idx,data in enumerate(tqdm(eval_data)):
                batch = copy.deepcopy(data)
                self.wash(batch)

                output= self.model1(batch,context_num=context_num)

                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()

                # visualize generated traffic snapshots
                if vis:
                    output_path = os.path.join('./vis/initialized',f'{idx}')
                    draw(center, output['agent'], other=rest, edge=bound, save=True,
                         path=output_path)

                agent,agent_mask = WaymoAgent.from_list_to_array(output['agent'])


                # save temp data for trafficgen to generate trajectory
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].cpu().numpy()
                output = {}
                output['context_num'] = context_num
                output['all_agent'] = agent
                output['agent_mask'] = agent_mask
                output['lane'] = batch['other']['lane'][0].cpu().numpy()
                output['unsampled_lane'] = batch['other']['unsampled_lane'][0].cpu().numpy()
                output['traf'] = self.eval_init_loader.dataset[idx]['other']['traf']
                output['gt_agent'] = batch['other']['gt_agent'][0].cpu().numpy()
                output['gt_agent_mask'] = batch['other']['gt_agent_mask'][0].cpu().numpy()

                p = os.path.join(save_path, f'{idx}.pkl')
                with open(p, 'wb') as f:
                    pickle.dump(output, f)

        return

    def generate_traj(self, snapshot=False,gif=False):
        if not os.path.exists('./vis/snapshots'):
            os.mkdir('./vis/snapshots')
        if not os.path.exists('./vis/gif'):
            os.mkdir('./vis/gif')

        self.model2.eval()
        cnt=0
        with torch.no_grad():
            pred_list = []

            for i in tqdm(range(self.cfg['data_usage'])):
                with open(f'./cases/initialized/{i}.pkl', 'rb+') as f:
                    data = pickle.load(f)

                pred_i = self.inference_control(data)

                pred_list.append(pred_i)

                if snapshot:

                    dir_path = f'./vis/snapshots/{i}'
                    ind = list(range(0,190,10))
                    agent = pred_i[ind]

                    agent_0 = agent[0]
                    agent0_list = []
                    agent_num = agent_0.shape[0]
                    for a in range(agent_num):
                        agent0_list.append(WaymoAgent(agent_0[[a]]))

                    cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(data['lane'][np.newaxis],
                                                                                    [data['traf'][0]],
                                                                                    center_num=1000, edge_num=500,
                                                                                    offest=0, lane_range=60)
                    draw_seq(cent[0],agent0_list,agent[...,:2],edge=bound[0],other=rest[0],path=dir_path,save=True)

                if gif:
                    dir_path = f'./vis/gif/{cnt}'
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)

                    ind = list(range(0,190,5))
                    agent = pred_i[ind]
                    agent = np.delete(agent,[2],axis=1)
                    for t in range(agent.shape[0]):
                        agent_t = agent[t]
                        agent_list = []
                        for a in range(agent_t.shape[0]):
                            agent_list.append(WaymoAgent(agent_t[[a]]))

                        path = os.path.join(dir_path, f'{t}')
                        cent,cent_mask,bound,bound_mask,_,_,rest,_ = process_map(data['lane'][np.newaxis],[data['traf'][int(t*5)]], center_num=2000, edge_num=1000,offest=0, lane_range=80,rest_num=1000)
                        draw(cent[0],agent_list,edge=bound[0],other=rest[0],path=path,save=True,vis_range=80)

                    images = []
                    for j in (range(38)):
                        file_name = os.path.join(dir_path, f'{j}.png')
                        img = imageio.imread(file_name)
                        images.append(img)

                    output = os.path.join(f'./vis/gif/{cnt}', f'movie_{i}.gif')
                    imageio.mimsave(output, images, duration=0.15)

                    cnt+=1

                    # if t==0:
                    #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                    #     for k in range(1,agent_t.shape[0]):
                    #         heat_path = os.path.join(dir_path, f'{k-1}')
                    #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)
            # if save_path:
            #     self.save_as_metadrive_data(pred_list,scene_data,save_path)

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

            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    batch[key] = batch[key].cuda()
            if self.cfg['device'] == 'cuda':
                self.model.cuda()

            pred = self.model2(batch,False)
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

        return pred_agent
