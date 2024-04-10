import copy
import os
import pickle

import imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trafficgen.act.model.tg_act import actuator
from trafficgen.init.model.tg_init import initializer
from trafficgen.init.utils.init_dataset import WaymoAgent
from trafficgen.traffic_generator.utils.data_utils import InitDataset, save_as_metadrive_data, from_list_to_batch, \
    transform_to_agent, process_case_to_input
from trafficgen.traffic_generator.utils.vis_utils import draw, draw_seq
from trafficgen.utils.utils import process_map, rotate


TRAFFICGEN_ROOT = os.path.dirname(os.path.dirname(__file__))

class TrafficGen:
    def __init__(self, cfg):
        self.cfg = cfg
        self.init_model = initializer.load_from_checkpoint(os.path.join(TRAFFICGEN_ROOT, "traffic_generator", "ckpt", "init.ckpt"))



        # act = actuator()
        # state = torch.load('traffic_generator/ckpt/act.ckpt', map_location='cpu')
        # act = torch.nn.DataParallel(act, device_ids=[0])
        # act.load_state_dict(state["state_dict"])
        # self.act_model = act
        self.act_model = actuator.load_from_checkpoint(os.path.join(TRAFFICGEN_ROOT, "traffic_generator", "ckpt", "act.ckpt"))

        use_cuda = torch.cuda.is_available()


        init_dataset = InitDataset(cfg)
        self.data_loader = DataLoader(init_dataset, shuffle=False, batch_size=1, num_workers=0)

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

    def generate_scenarios(self, gif=True, save_metadrive=False):
        print('Initializing traffic scenarios...')
        self.place_vehicles(vis=True)
        print('Complete.\n' 'Visualization results are saved in traffic_generator/output/vis/scene_initialized\n')

        print('Generating trajectories...')
        self.generate_traj(snapshot=True, gif=gif, save_metadrive=save_metadrive)
        print('Complete.\n' 'Visualization results are saved in traffic_generator/output/vis/scene_static\n')

    def place_vehicles_for_single_scenario(self, batch, index=None, vis=False, vis_dir=None, context_num=1):
        self.wash(batch)

        # Call the initialization model
        # The output is a dict with these keys: center, rest, bound, agent
        model_output = self.init_model.inference(batch, context_num=context_num)

        center = batch['center'][0].cpu().numpy()
        rest = batch['rest'][0].cpu().numpy()
        bound = batch['bound'][0].cpu().numpy()

        # visualize generated traffic snapshots
        if vis:
            assert vis_dir is not None
            assert index is not None
            output_path = os.path.join(vis_dir, f'{index}')
            draw(center, model_output['agent'], other=rest, edge=bound, save=True, path=output_path)

        return model_output

    def place_vehicles(self, vis=True):
        context_num = 1

        init_vis_dir = 'traffic_generator/output/vis/scene_initialized'
        if not os.path.exists(init_vis_dir):
            os.makedirs(init_vis_dir)
        tmp_pth = 'traffic_generator/output/initialized_tmp'
        if not os.path.exists(tmp_pth):
            os.makedirs(tmp_pth)

        self.init_model.eval()

        data_path = self.cfg['data_path']
        with torch.no_grad():
            for idx, data in enumerate(tqdm(self.data_loader)):
                data_file_path = os.path.join(data_path, f'{idx}.pkl')
                with open(data_file_path, 'rb+') as f:
                    original_data = pickle.load(f)

                batch = copy.deepcopy(data)

                model_output = self.place_vehicles_for_single_scenario(batch, idx, vis, init_vis_dir, context_num)

                agent, agent_mask = WaymoAgent.from_list_to_array(model_output['agent'])

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
                output['traf'] = self.data_loader.dataset[idx]['other']['traf']
                output['gt_agent'] = batch['other']['gt_agent'][0].cpu().numpy()
                output['gt_agent_mask'] = batch['other']['gt_agent_mask'][0].cpu().numpy()

                if "center_info" in original_data:
                    output['center_info'] = original_data['center_info']
                else:
                    output["center_info"] = {}

                p = os.path.join(tmp_pth, f'{idx}.pkl')
                with open(p, 'wb') as f:
                    pickle.dump(output, f)

        return

    def generate_traj(self, snapshot=True, gif=False, save_metadrive=False):
        snapshot_path = 'traffic_generator/output/vis/scene_static'
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        gif_path = 'traffic_generator/output/vis/scene_gif'
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)
        pkl_path = 'traffic_generator/output/scene_pkl'
        if not os.path.exists(pkl_path):
            os.makedirs(pkl_path)

        self.act_model.eval()

        with torch.no_grad():
            for i in tqdm(range(self.cfg['data_usage'])):
                with open(f'traffic_generator/output/initialized_tmp/{i}.pkl', 'rb+') as f:
                    data = pickle.load(f)
                pred_i = self.inference_control(data)
                if snapshot:
                    ind = list(range(0, 190, 10))
                    agent = pred_i[ind]

                    agent_0 = agent[0]
                    agent0_list = []
                    agent_num = agent_0.shape[0]
                    for a in range(agent_num):
                        agent0_list.append(WaymoAgent(agent_0[[a]]))

                    cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(
                        data['lane'][np.newaxis], [data['traf'][0]],
                        center_num=1000,
                        edge_num=500,
                        offest=0,
                        lane_range=60
                    )
                    img_path = os.path.join(snapshot_path, f'{i}')
                    draw_seq(
                        cent[0], agent0_list, agent[..., :2], edge=bound[0], other=rest[0], path=img_path, save=True
                    )

                if gif:
                    dir_path = os.path.join(gif_path, f'{i}')
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)

                    ind = list(range(0, 190, 5))
                    agent = pred_i[ind]
                    if agent.shape[1] > 2:  # PZH: I don't know what this means here. Just add this if to avoid error.
                        agent = np.delete(agent, [2], axis=1)
                    for t in range(agent.shape[0]):
                        agent_t = agent[t]
                        agent_list = []
                        for a in range(agent_t.shape[0]):
                            agent_list.append(WaymoAgent(agent_t[[a]]))

                        path = os.path.join(dir_path, f'{t}')
                        cent, cent_mask, bound, bound_mask, _, _, rest, _ = process_map(
                            data['lane'][np.newaxis], [data['traf'][int(t * 5)]],
                            center_num=2000,
                            edge_num=1000,
                            offest=0,
                            lane_range=80
                        )
                        draw(cent[0], agent_list, edge=bound[0], other=rest[0], path=path, save=True, vis_range=80)

                    images = []
                    for j in (range(38)):
                        file_name = os.path.join(dir_path, f'{j}.png')
                        img = imageio.imread(file_name)
                        images.append(img)
                    output = os.path.join(gif_path, f'movie_{i}.gif')
                    imageio.mimsave(output, images, duration=0.1)

                    # if t==0:
                    #     center, _, bounder, _, _, _, rester = WaymoDataset.process_map(inp, 2000, 1000, 50,0)
                    #     for k in range(1,agent_t.shape[0]):
                    #         heat_path = os.path.join(dir_path, f'{k-1}')
                    #         draw(cent, heat_map[k-1], agent_t[:k], rest, edge=bound, save=True, path=heat_path)

                if save_metadrive:
                    data_dir = os.path.join(pkl_path, f'{i}.pkl')
                    save_as_metadrive_data(i, pred_i, data, data_dir)

        if gif:
            print("GIF files have been generated to vis/gif folder.")
        if snapshot:
            print("Trajectory visualization has been generated to vis/snapshots folder.")
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
            pred_agent[:, 0, :7] = future_traj[:190, 0]
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

            pred = self.act_model(batch)
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
