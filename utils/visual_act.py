import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams.update({'figure.max_open_warning': 0})
from l5kit.configs import load_config_data
import torch
import matplotlib.colors as mcolors

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


def draw_real(ax,center, edge, rest, context,path=None,save=False,gt=False):
    colors = list(mcolors.TABLEAU_COLORS)

    #plt.axis('equal')
    # lim = ax.viewLim
    # lim.x0,lim.y0,lim.x1,lim.y1 = -50,-50,50,50
    # ax.set_clip_box(lim)
    vl = 70
    # for j in range(center.shape[0]):
    #     traf_state = center[j, -1]
    #     x0, y0, x1, y1, = center[j, :4]
    #     if x0 == 0: break
    #     if traf_state == 1:
    #         color = 'red'
    #         ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=0.5, zorder=5000)
    #     elif traf_state == 2:
    #         color = 'yellow'
    #         ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=0.5, zorder=5000)
    #     elif traf_state == 3:
    #         color = 'green'
    #         ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=0.5, zorder=5000)

    # hist_traj = context[:,:-1,:2]
    # context = context[:,-1]
    # for i in range(hist_traj.shape[1]):
    #     all_agent = hist_traj[:,i]
    #     valid = (abs(all_agent[:,0])<vl)*(abs(all_agent[:,1])<vl)
    #     all_agent = all_agent[valid]
    #     ax.scatter(all_agent[:,0],all_agent[:,1],s=0.3,c='royalblue',alpha=0.015*i,marker='.',edgecolors='none')

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            if abs(x0)>vl or abs(x1)>vl or abs(y0)>vl or abs(y1)>vl: continue
            ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.1)

    if rest is not None:
        for j in range(len(rest)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = rest[j, :4]
            if x0 == 0: break
            if abs(x0) > vl or abs(x1) > vl or abs(y0) > vl or abs(y1) > vl: continue
            ax.plot((x0, x1), (y0, y1), 'grey', linewidth=0.05)

    if gt:
        for i in range(context.shape[0]):
            ind = i % 10
            col = colors[ind]
            agent = context[i]
            center = agent[:2]
            vel = agent[2:4]
            yaw = np.arctan2(agent[6], agent[7])
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
                                 facecolor=col, linewidth=0.04)

            ax.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-r', linewidth=0.04, markersize=0.2,markeredgecolor='none')
            rect.set_zorder(10000)
            ax.add_patch(rect)
    else:
        for i in range(context.shape[0]):
            ind = i%10
            col = colors[ind]
            agent = context[i]
            center = agent[:2]
            if abs(center[0])>vl or abs(center[1])>vl:
                continue
            vel = agent[2:4]
            yaw = -agent[4]-np.pi/2
            L, W = agent[5:7]
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
                                 facecolor=col, linewidth=0.04)
            ax.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-r', linewidth=0.04, markersize=0.2,markeredgecolor='none')
            ax.add_patch(rect)



def draw_seq(center, edge, rest, context,path=None,save=False,gt=False):
    colors = list(mcolors.TABLEAU_COLORS)
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')
    # lim = ax.viewLim
    # lim.x0,lim.y0,lim.x1,lim.y1 = -50,-50,50,50
    # ax.set_clip_box(lim)
    vl = 60
    ax.axis('off')
    for j in range(center.shape[0]):
        traf_state = center[j, -1]
        x0, y0, x1, y1, = center[j, :4]
        if x0 == 0: break
        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=5, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=5, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1, linewidth=5, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.5)

    if rest is not None:
        for j in range(len(rest)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = rest[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), 'grey', linewidth=0.5)

    if gt:
        for i in range(context.shape[0]):
            ind = i % 10
            col = colors[ind]
            agent = context[i]
            center = agent[:2]
            vel = agent[2:4]
            yaw = np.arctan2(agent[6], agent[7])
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
                                 facecolor=col, linewidth=0.2)
            plt.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-r', linewidth=0.5,
                     markersize=1, zorder=100000)
            rect.set_zorder(10000)
            ax.add_patch(rect)
    else:
        for i in range(context.shape[0]):
            ind = i%10
            col = colors[ind]
            agent = context[i]
            center = agent[:2]
            if abs(center[0])>vl or abs(center[1])>vl:
                continue
            vel = agent[2:4]
            yaw = -agent[4]-np.pi
            #yaw = -agent[4]
            L, W = agent[5:7]
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
                                 facecolor=col, linewidth=0.2)
            if abs(center[0])<vl and abs(vel[0] + center[0])<vl and abs(center[1])<vl and abs(vel[1] + center[1])<vl:
                ax.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-r', linewidth=0.5, markersize=1)
            rect.set_zorder(0)
            ax.add_patch(rect)

    if save:
        fig.savefig(path, dpi=100,bbox_inches='tight',pad_inches=0)



# def draw(center, context, cnt=0,agents=None, save=False, edge=None,rest=None, path='../vis'):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     plt.axis('equal')
#     #ax.axis('off')
#
#     for j in range(center.shape[0]):
#         traf_state = center[j,-1]
#         x0, y0, x1, y1, = center[j, :4]
#         if x0 == 0:break
#         if traf_state==1:
#             color = 'red'
#             ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1,linewidth=5,zorder=5000)
#         elif traf_state==2:
#             color = 'yellow'
#             ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1,linewidth=5,zorder=5000)
#         elif traf_state==3:
#             color ='green'
#             ax.plot((x0, x1), (y0, y1), color=color, alpha=0.1,linewidth=5,zorder=5000)
#
#     if edge is not None:
#         for j in range(len(edge)):
#
#             # if lane[j, k, -1] == 0: continue
#             x0, y0, x1, y1, = edge[j,  :4]
#             if x0 == 0:break
#             ax.plot((x0, x1), (y0, y1), 'black', linewidth=0.5)
#
#     if rest is not None:
#         for j in range(len(rest)):
#
#             # if lane[j, k, -1] == 0: continue
#             x0, y0, x1, y1, = rest[j,  :4]
#             if x0 == 0:break
#             ax.plot((x0, x1), (y0, y1), 'grey', linewidth=0.5)
#
#
#
#     for i in range(context.shape[0]):
#         agent = context[i]
#         center = agent[:2]
#         vel = agent[2:4]
#         yaw = np.arctan2(agent[6],agent[7])
#         L, W = agent[4:6]
#         l, w = L / 2, W / 2
#         x1 = w / np.cos(yaw)
#         x2 = x1 * np.sin(yaw)
#         x3 = l - x2
#         x4 = x3 * np.sin(yaw)
#         x_ = x1 + x4
#         y_ = x3 * np.cos(yaw)
#         point_x = center[0] - x_
#         point_y = center[1] - y_
#         rect = plt.Rectangle([point_x, point_y], W, L, -yaw * 180 / np.pi, edgecolor="black",
#                              facecolor="royalblue", linewidth=0.2)
#         plt.plot([center[0], vel[0]+center[0]], [center[1], vel[1]+center[1]],'.-r',linewidth=0.5,markersize=1,zorder=100000)
#         rect.set_zorder(10000)
#         ax.add_patch(rect)
#
#     if agents is not None:
#         t,a,_ = agents.shape
#         for i in range(a):
#             agent_pos = agents[:,i,:2]
#             mask = (abs(agent_pos[:,0])<100) * (abs(agent_pos[:,1])<100)
#             agent_pos= agent_pos[mask]
#
#             ax.plot(agent_pos[:,0],agent_pos[:,1],'--',color='royalblue')
#
#     plt.autoscale()
#     if save:
#         pa = os.path.join(path)
#         fig.savefig(pa, dpi=1000)
#
#     return plt

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


