import os
from drivingforce.TrafficTranformer.data_process.map_data_process import WaymoMap
from drivingforce.TrafficTranformer.data_process.agent_process import WaymoScene, WaymoAgentInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from tqdm import tqdm
import matplotlib
from utils.dataLoader import WaymoDataset
matplotlib.rcParams.update({'figure.max_open_warning': 0})
from l5kit.configs import load_config_data
import torch

def draw(lane, agents, cnt=0, save=False, edge=None,path='./vis',abn_idx=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')

    lane = lane.detach().numpy()
    agents = agents.detach().numpy()

    # if taff_l is not None:
    #     for j in range(taff_l.shape[0]):
    #         traf = taff_l[j]
    #         if traf[-1] == 0. or traf[-2] == 0.:
    #             plt.scatter(traf[1], traf[2])

    for j in range(lane.shape[0]):
        for k in range(lane.shape[1]):
            # color = "orange" if (lane[j, k, -2:] == np.array([1, 1])).all() else "grey"
            color = 'grey'
            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = lane[j, k, :4]
            ax.plot((x0, x1), (y0, y1), color, linewidth=0.3 if color=="grey" else 1)
            # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)

    if edge is not None:
        for j in range(len(edge)):
            for k in range(edge.shape[1]):
                # if lane[j, k, -1] == 0: continue
                x0, y0, x1, y1, = edge[j, k, :4]
                if x0 != x1 and y0 != y1 and x0 != 0 and y0 != 0 and x1 != 0 and y1 != 0:
                    ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.5)
                # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)

    for j in range(agents.shape[0]):
        if abn_idx is not None and abn_idx == j-1:
            cc = 'red'
        else:
            cc = 'royalblue'

        agent = WaymoAgentInfo(agents[j])

        center = agent.position
        end_p = agent.end_point
        yaw = agent.heading
        vx, vy = agent.velocity
        L, W = agent.length_width
        l, w = L / 2, W / 2
        x1 = w / np.cos(yaw)
        x2 = x1 * np.sin(yaw)
        x3 = l - x2
        x4 = x3 * np.sin(yaw)
        x_ = x1 + x4
        y_ = x3 * np.cos(yaw)
        point_x = center[0] - x_
        point_y = center[1] - y_
        if j == 0:
            rect = plt.Rectangle([point_x, point_y], W, L, -yaw * 180 / np.pi, edgecolor="black", facecolor="orangered",linewidth=0.2)
        else:
            rect = plt.Rectangle([point_x, point_y], W, L, -yaw * 180 / np.pi,edgecolor="black",facecolor=cc,linewidth=0.2)
        rect.set_zorder(0)
        ax.add_patch(rect)
        #plt.plot([center[0], vx+center[0]], [center[1], vy+center[1]],'.-r',linewidth=0.5,markersize=1)
        ax.plot([center[0],end_p[0]],[center[1],end_p[1]],'--',c=cc,linewidth=0.6,alpha=1)
        if not end_p[0] == center[0]:
            ax.scatter(end_p[0],end_p[1],marker='x',s=6,c='k',linewidths=0.5,alpha=0.7)
        else:
            ax.scatter(end_p[0], end_p[1], marker='.', s=0.5,linewidths=0.5, c='k')
        # plt.ylim(-50,50)
        # plt.xlim(-50.50)
        #
        # elif agent.type == 'ped':
        #     center = agent.position
        #     vx, vy = agent.velocity
        #     plt.scatter(center[0], center[1],color='g')
        #     ax.arrow(center[0], center[1], vx / 3, vy / 3)
        # elif agent.type == 'cyc':
        #     center = agent.position
        #     vx, vy = agent.velocity
        #     plt.scatter(center[0], center[1],color='r')
        #     ax.arrow(center[0], center[1], vx / 3, vy / 3)

    plt.autoscale()
    if save:
        pa = os.path.join(path,f'{cnt}')
        fig.savefig(pa, dpi=1000)

    return plt


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    cnt = 0

    with open('./ab_data.pkl', 'rb+') as f:
        abn = pickle.load(f)

    tt_path = '/Users/fenglan/Downloads/waymo/scenes_tt'

    # sample = os.path.join(tt_path,'189.pkl')
    # with open(sample, 'rb+') as f:
    #     data = pickle.load(f)
    # agents = data['nbrs_p_c_f']

    #
    # lane1 = data['lane'][50]
    # lane2 = data['lane'][146]
    # fig, ax = plt.subplots(figsize=(10, 10))
    # for j in range(lane2.shape[0]):
    #     for k in range(lane2.shape[1]):
    #         # color = "orange" if (lane[j, k, -2:] == np.array([1, 1])).all() else "grey"
    #         color = 'grey'
    #         # if lane[j, k, -1] == 0: continue
    #         #x0, y0, x1, y1, = lane[j, k, :4]
    #         ax.scatter(lane2[j, k, 0],lane2[j,k,1],s=0.5)
    #         #ax.plot((x0, x1), (y0, y1), color, linewidth=0.3 if color=="grey" else 1)
    # for i in range(agents.shape[0]):
    #     if agents[i,146,-1]:
    #         ax.scatter(agents[i,146,0],agents[i,146,1])

    cfg_path = './cfg/debug.yaml'

    cfg = load_config_data(cfg_path)
    cfg['eval_data_usage'] = 300
    cfg['eval_data_start_index'] = 0
    cfg['data_path'] = tt_path
    train_dataset = WaymoDataset(cfg,eval=True)
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0)

   # path = "./debug_data"
    cnt = 0
    for i in range(10):
        data = train_dataset[abn['abn_seq'][i]]
        ts=8
        draw(data["rest"][ts], data['agent'][ts], cnt=cnt,
             edge=data["bound"][ts], save=True,path='./gt',abn_idx=abn['idx'][i])
        cnt+=1


    # cnt=0
    # for data in tqdm(data_loader):
    #     for key in data.keys():
    #         if isinstance(data[key], torch.DoubleTensor):
    #             data[key] = data[key].float()
    #     ts=8
    #     draw(data["rest"][0][ts], data['agent'][0][ts], cnt=cnt,
    #          edge=data["bound"][0][ts], save=True,path='./gt')
    #     cnt+=1
