import copy
import os
import pickle

import imageio
import numpy as np
import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from kmeans_pytorch import kmeans
from sklearn import manifold

from trafficgen.TrafficGen_act.data_process.act_dataset import process_case_to_input, process_map
from trafficgen.TrafficGen_act.models.act_model import Actuator
from trafficgen.TrafficGen_init.data_process.init_dataset import initDataset, WaymoAgent
from trafficgen.TrafficGen_init.models.init_cluster import Initializer
from trafficgen.utils.utils import transform_to_agent, from_list_to_batch, rotate, save_as_metadrive_data
from trafficgen.utils.visual_init import draw, draw_seq

from utils.tsne import visualize_tsne_points,visualize_tsne_images

import matplotlib as plt
#from trafficgen.utils.tsne import tsne

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
        wandb.init(project="cluster")
        model1 = Initializer(cfg['init_model'])
        model2 = Actuator()
        model1 = model1.to(self.cfg['device'])
        model2 = model2.to(self.cfg['device'])
        devices = [0]
        model1 = torch.nn.DataParallel(model1, device_ids=devices)
        model2 = torch.nn.DataParallel(model2, device_ids=devices)

        self.model1 = model1
        self.model2 = model2

        test_init_dataset = initDataset(self.cfg)
        self.eval_init_loader = DataLoader(test_init_dataset, shuffle=False, batch_size=1, num_workers=0)

    def load_model(self, model, model_path, device):
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

    def wash(self, batch):
        """Transform the loaded raw data to pretty pytorch tensor."""
        for key in batch.keys():
            if isinstance(batch[key], np.ndarray):
                batch[key] = Tensor(batch[key])
            if isinstance(batch[key], torch.DoubleTensor):
                batch[key] = batch[key].float()
            if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                batch[key] = batch[key].cuda()
            if 'mask' in key:
                batch[key] = batch[key].to(bool)

    def generate_scenarios(self, snapshot=True, gif=True, save_pkl=False):
        # generate temp data in ./cases/initialized, and visualize in ./vis/initialized
        self.place_vehicles(vis=True)

        # generate trajectory from temp data, and visualize in ./vis/snapshots.
        # set gif to True to generate gif in ./vis/gif
        self.generate_traj(snapshot=snapshot, gif=gif, save_pkl=save_pkl)

    def tsne(self):
        vis_num = 100

        self.model1.eval()
        eval_data = self.eval_init_loader
        datasize = len(eval_data)

        features = torch.zeros([datasize,1024],device=self.cfg['device'])

        ret = {}
        with torch.no_grad():
            for idx, data in enumerate(tqdm(eval_data)):

                batch = copy.deepcopy(data)
                self.wash(batch)
                features[idx]=self.model1(batch, context_num=0)

            tsne = manifold.TSNE(
                n_components=2,
                init='pca',
                learning_rate='auto',
                n_jobs=-1
            )

            Y = tsne.fit_transform(features.numpy())

            rand_indx = list(range(datasize))
            np.random.shuffle(rand_indx)

            Y = Y[rand_indx]
            sampled_indx = rand_indx[:vis_num]
            ret['tsne_points'] = wandb.Image(visualize_tsne_points(Y[:1000]))


            img_path = './img'
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            for i in range(vis_num):

                data = eval_data.dataset[sampled_indx[i]]
                agent = data['agent']
                agent_list = []
                agent_num = agent.shape[0]
                for a in range(agent_num):
                    agent_list.append(WaymoAgent(agent[[a]]))

                draw(data['center'], agent_list, other=data['rest'], edge=data['bound'],path=f'./img/{i}.jpg',save=True)

            image_path = [f'./img/{i}.jpg' for i in range(vis_num)]
            ret['tsne_image'] = visualize_tsne_images(Y[:vis_num,0],Y[:vis_num,1],image_path)

        wandb.log(ret)

    def cluster(self):
        self.model1.eval()
        eval_data = self.eval_init_loader
        datasize = len(eval_data)
        num_clusters = 40

        features = torch.zeros([datasize,1024],device=self.cfg['device'])
        with torch.no_grad():
            for idx, data in enumerate(tqdm(eval_data)):

                batch = copy.deepcopy(data)
                self.wash(batch)
                features[idx]=self.model1(batch, context_num=0)

        cluster_ids_x, cluster_centers = kmeans(
            X=features, num_clusters=num_clusters, distance='euclidean', device=self.cfg['device'])

        # group = {}
        # for i in range(num_clusters):
        #     idx = torch.where(cluster_ids_x==i)
        #     group[i] = features[idx]
        ret = {}
        for k in range(5):
            idxs = torch.where(cluster_ids_x == k)[0]
            features_ = features[idxs]
            dist_to_first = torch.nn.MSELoss(reduction='none')(features_[[0]],features_).mean(-1)
            dist_indx = torch.argsort(dist_to_first)

            for i in range(10):
                dist_indx_i = dist_indx[i]
                idxs_i = idxs[dist_indx_i].item()
                data = eval_data.dataset[idxs_i]

                agent = data['agent']
                agent_list = []
                agent_num = agent.shape[0]
                for a in range(agent_num):
                    agent_list.append(WaymoAgent(agent[[a]]))

                ret[f'{i+k*10}']=wandb.Image(draw(data['center'], agent_list, other=data['rest'], edge=data['bound']))


        wandb.log(ret)
    def place_vehicles(self, vis=True):
        context_num = 1

        vis_path = './vis/initialized'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        save_path = './cases/initialized'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model1.eval()
        eval_data = self.eval_init_loader
        data_path = self.cfg['data_path']
        with torch.no_grad():
            for idx, data in enumerate(tqdm(eval_data)):
                data_file_path = os.path.join(data_path, f'{idx}.pkl')
                with open(data_file_path, 'rb+') as f:
                    original_data = pickle.load(f)

                batch = copy.deepcopy(data)
                self.wash(batch)

                # Call the initialization model
                # The output is a dict with these keys: center, rest, bound, agent
                output = self.model1(batch, context_num=context_num)

                center = batch['center'][0].cpu().numpy()
                rest = batch['rest'][0].cpu().numpy()
                bound = batch['bound'][0].cpu().numpy()

                # visualize generated traffic snapshots
                if vis:
                    output_path = os.path.join('./vis/initialized', f'{idx}')
                    draw(center, output['agent'], other=rest, edge=bound, save=True,
                         path=output_path)

                agent, agent_mask = WaymoAgent.from_list_to_array(output['agent'])

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
                output['center_info'] = original_data['center_info']

                p = os.path.join(save_path, f'{idx}.pkl')
                with open(p, 'wb') as f:
                    pickle.dump(output, f)

        return

    def generate_traj(self, snapshot=True, gif=False, save_pkl=False):
        if not os.path.exists('./vis/snapshots'):
            os.mkdir('./vis/snapshots')
        if not os.path.exists('./vis/gif'):
            os.mkdir('./vis/gif')

        self.model2.eval()

        with torch.no_grad():

            for i in tqdm(range(self.cfg['data_usage'])):
                with open(f'./cases/initialized/{i}.pkl', 'rb+') as f:
                    data = pickle.load(f)

                pred_i = self.inference_control(data)

                if snapshot:

                    dir_path = f'./vis/snapshots/{i}'
                    ind = list(range(0, 190, 10))
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
                    draw_seq(cent[0], agent0_list, agent[..., :2], edge=bound[0], other=rest[0], path=dir_path,
                             save=True)

                if gif:
                    dir_path = f'./vis/gif/{i}'
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)

                    ind = list(range(0, 190, 5))
                    agent = pred_i[ind]
                    agent = np.delete(agent, [2], axis=1)
                    for t in range(agent.shape[0]):
                        agent_t = agent[t]
                        agent_list = []
                        for a in range(agent_t.shape[0]):
                            agent_list.append(WaymoAgent(agent_t[[a]]))

                        path = os.path.join(dir_path, f'{t}')
                        cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(data['lane'][np.newaxis],
                                                                                        [data['traf'][int(t * 5)]],
                                                                                        center_num=2000, edge_num=1000,
                                                                                        offest=0, lane_range=80,
                                                                                        rest_num=1000)
                        draw(cent[0], agent_list, edge=bound[0], other=rest[0], path=path, save=True, vis_range=80)

                    images = []
                    for j in (range(38)):
                        file_name = os.path.join(dir_path, f'{j}.png')
                        img = imageio.imread(file_name)
                        images.append(img)
                    output = os.path.join(f'./vis/gif', f'movie_{i}.gif')
                    imageio.mimsave(output, images, duration=0.1)

                    # if t==0:
                    #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                    #     for k in range(1,agent_t.shape[0]):
                    #         heat_path = os.path.join(dir_path, f'{k-1}')
                    #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)

                if save_pkl:
                    if not os.path.exists(f'./generated_scenarios'):
                        os.makedirs(f'./generated_scenarios')
                    data_dir = f'./generated_scenarios/{i}.pkl'
                    other = {}
                    other['unsampled_lane'] = data['unsampled_lane']
                    other['center_info'] = data['center_info']
                    save_as_metadrive_data(pred_i, other, data_dir)

        if gif:
            print("GIF files have been generated to vis/gif folder.")
        if snapshot:
            print("Trajectory visualization has been generated to vis/snapshots folder.")
        if save_pkl:
            print('Generated scenarios have been saved to generated_scenarios folder')

    def inference_control(self, data, ego_gt=True, length=190, per_time=20):
        # for every x time step, pred then update
        agent_num = data['agent_mask'].sum()
        data['agent_mask'] = data['agent_mask'][:agent_num]
        data['all_agent'] = data['all_agent'][:agent_num]

        pred_agent = np.zeros([length, agent_num, 8])
        pred_agent[0, :, :7] = copy.deepcopy(data['all_agent'])
        pred_agent[1:, :, 5:7] = pred_agent[0, :, 5:7]

        start_idx = 0

        if ego_gt == True:
            future_traj = data['gt_agent']
            pred_agent[:, 0, :7] = future_traj[:, 0]
            start_idx = 1

        for i in range(0, length - 1, per_time):

            current_agent = copy.deepcopy(pred_agent[i])
            case_list = []
            for j in range(agent_num):
                a_case = {}
                a_case['agent'], a_case['lane'] = transform_to_agent(current_agent[j], current_agent, data['lane'])
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

            pred = self.model2(batch, False)
            prob = pred['prob']
            velo_pred = pred['velo']
            pos_pred = pred['pos']
            heading_pred = pred['heading']
            all_pred = torch.cat([pos_pred, velo_pred, heading_pred.unsqueeze(-1)], dim=-1)

            best_pred_idx = torch.argmax(prob, dim=-1)
            best_pred_idx = best_pred_idx.view(agent_num, 1, 1, 1).repeat(1, 1, *all_pred.shape[2:])
            best_pred = torch.gather(all_pred, dim=1, index=best_pred_idx).squeeze(1).cpu().numpy()

            ## update all the agent
            for j in range(start_idx, agent_num):
                pred_j = best_pred[j]
                agent_j = copy.deepcopy(current_agent[j])
                center = copy.deepcopy(agent_j[:2])
                center_yaw = copy.deepcopy(agent_j[4])

                pos = rotate(pred_j[:, 0], pred_j[:, 1], center_yaw)
                heading = pred_j[..., -1] + center_yaw
                vel = rotate(pred_j[:, 2], pred_j[:, 3], center_yaw)

                pos = pos + center
                pad_len = pred_agent[i + 1:i + per_time + 1].shape[0]
                pred_agent[i + 1:i + per_time + 1, j, :2] = copy.deepcopy(pos[:pad_len])
                pred_agent[i + 1:i + per_time + 1, j, 2:4] = copy.deepcopy(vel[:pad_len])
                pred_agent[i + 1:i + per_time + 1, j, 4] = copy.deepcopy(heading[:pad_len])

        return pred_agent
