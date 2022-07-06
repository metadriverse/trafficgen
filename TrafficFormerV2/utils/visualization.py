import os
from TrafficFormerV2.data_process.map_data_process import WaymoMap
from TrafficFormerV2.data_process.agent_process import WaymoScene, WaymoAgentInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from tqdm import tqdm
import matplotlib
from TrafficFormerV2.utils.dataLoader import WaymoDataset
matplotlib.rcParams.update({'figure.max_open_warning': 0})
from l5kit.configs import load_config_data
import torch
from TrafficFormerV2.utils.dataLoader import WaymoDataset
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm

def transform_coord(x,y, angle):

    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)

    return output_coords

def draw_heatmap(vector,vector_prob,gt_idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    vector_prob = vector_prob.cpu().numpy()

    for j in range(vector.shape[0]):
        if j in gt_idx: color = (0,0,1)
        else:
            grey_scale = max(0,0.9-vector_prob[j])
            color = (0.9,grey_scale,grey_scale)

        # if lane[j, k, -1] == 0: continue
        x0, y0, x1, y1, = vector[j, :4]
        ax.plot((x0, x1), (y0, y1),color=color, linewidth=2)

    return plt



def draw(center, heat_map,agents, other,cnt=0, save=False, edge=None,path='../vis',abn_idx=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')
    # ax.set_xlim(heat_map[1][:2])
    # ax.set_ylim(heat_map[1][2:])


    ax.imshow(heat_map[0], extent=heat_map[1],alpha=1, origin='lower', cmap=cm.jet)
    ax.axis('off')

    for j in range(center.shape[0]):
        traf_state = center[j,-1]
        #prob = center[j,-1]
        x0, y0, x1, y1, = center[j, :4]


        #grey_scale = max(0,0.96-prob)
        #color = (grey_scale,grey_scale,0.96)
        #color = (prob, 0, (1-prob)*0.5)

        if x0 == 0:break
        #order = int(prob*100)
        ax.plot((x0, x1), (y0, y1),'--', color='white', linewidth=2,alpha=0.2)

        if traf_state==1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.5,linewidth=3,zorder=5000)
        elif traf_state==2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.5,linewidth=3,zorder=5000)
        elif traf_state==3:
            color ='green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.5,linewidth=3,zorder=5000)

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j,  :4]
            if x0 == 0:break
            ax.plot((x0, x1), (y0, y1), 'white', linewidth=1)
            # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)
    if other is not None:
        for j in range(len(other)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = other[j,  :4]
            if x0 == 0:break
            ax.plot((x0, x1), (y0, y1), 'white', linewidth=0.7)

    a,dim = agents.shape

    for i in range(a):
        agent = agents[i]
        center = agent[:2]
        vel = agent[2:4]
        yaw = np.arctan2(agent[6],agent[7])
        L, W = agent[4:6]
        l, w = L / 2, W / 2
        x1 = w / np.cos(yaw)
        x2 = x1 * np.sin(yaw)
        x3 = l - x2
        x4 = x3 * np.sin(yaw)
        x_ = x1 + x4
        y_ = x3 * np.cos(yaw)
        point_x = center[0] - x_
        point_y = center[1] - y_
        rect = plt.Rectangle([point_x, point_y], W, L, -yaw * 180 / np.pi, edgecolor="black",
                             facecolor="snow", linewidth=1,zorder=10000)
        ax.plot([center[0], vel[0]+center[0]], [center[1], vel[1]+center[1]],'.-',color='lime',linewidth=1.5,markersize=2.5,zorder=10000)
        #rect.set_zorder(1000)
        ax.add_patch(rect)

    # for i in range(agents.shape[0]):
    #     agent = agents[i]
    #     center = agent[:2]
    #     if abs(center[0])>100 or abs(center[1]-25)>80:
    #         continue
    #     vel = agent[2:4]
    #     yaw = -agent[4]-np.pi
    #     L, W = agent[5:7]
    #     l, w = L / 2, W / 2
    #     x1 = w / np.cos(yaw)
    #     x2 = x1 * np.sin(yaw)
    #     x3 = l - x2
    #     x4 = x3 * np.sin(yaw)
    #     x_ = x1 + x4
    #     y_ = x3 * np.cos(yaw)
    #     point_x = center[0] - x_
    #     point_y = center[1] - y_
    #     rect = plt.Rectangle([point_x, point_y], W, L, -yaw * 180 / np.pi, edgecolor="black",
    #                          facecolor="snow", linewidth=1,zorder=10000)
    #     ax.plot([center[0], vel[0]+center[0]], [center[1], vel[1]+center[1]],'.-',color='lime',linewidth=1.5,markersize=2.5,zorder=10000)
    #     #rect.set_zorder(1000)
    #     ax.add_patch(rect)


    plt.xlim(heat_map[1][:2])
    plt.ylim(heat_map[1][2:])
    plt.autoscale()
    if save:
        fig.savefig(path, dpi=100,bbox_inches='tight',pad_inches=0)

    return plt

def get_agent_pos_from_vec(vec,long_perc,lat_perc,dir, v_value):
    x1,y1,x2,y2 = vec[:,0],vec[:,1],vec[:,2],vec[:,3]
    vec_len = ((x1-x2)**2+(y1-y2)**2)**0.5

    vec_dir = -np.arctan2(y2 - y1, x2 - x1)+np.pi/2

    long_pos = vec_len*long_perc
    lat_pos = lat_perc*4

    coord = transform_coord(lat_pos,long_pos,-vec_dir)

    coord[:,0]+=x1
    coord[:,1]+=y1

    agent_dir = vec_dir+dir
    #v_dir=v_dir+agent_dir
    v_dir = agent_dir

    vel = np.stack([np.sin(v_dir)*v_value,np.cos(v_dir)*v_value], axis=-1)

    dir_ = np.stack([np.sin(agent_dir),np.cos(agent_dir)], axis=-1)
    return coord, dir_,vel


if __name__ == "__main__":
    cfg_path = '../cfg/debug.yaml'


    cfg = load_config_data(cfg_path)
    train_dataset = WaymoDataset(cfg, eval=True)

    cnt = 0
    for i in range(0,10):
        data = train_dataset[i]
        mask = data['agent_mask']
        vec_ind = data['vec_index'].numpy().astype(int)
        vec = data['center'][vec_ind]

        for key in data.keys():

            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].numpy()

        agent_pos, agent_dir, agent_vel = get_agent_pos_from_vec(vec,data['long_perc'],data['lat_perc'],data['relative_dir'], data['v_value'])
        draw(data['center'], data['agent'][mask.to(bool)], edge=data['bound'], cnt=cnt,save=True)
        cnt+=1


