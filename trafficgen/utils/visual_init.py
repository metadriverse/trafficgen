import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'figure.max_open_warning': 0})
from scipy.ndimage.filters import gaussian_filter
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def get_heatmap(x, y, prob, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=prob, density=True)

    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def draw_heatmap(vector, vector_prob, gt_idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    vector_prob = vector_prob.cpu().numpy()

    for j in range(vector.shape[0]):
        if j in gt_idx:
            color = (0, 0, 1)
        else:
            grey_scale = max(0, 0.9 - vector_prob[j])
            color = (0.9, grey_scale, grey_scale)

        # if lane[j, k, -1] == 0: continue
        x0, y0, x1, y1, = vector[j, :4]
        ax.plot((x0, x1), (y0, y1), color=color, linewidth=2)

    return plt


def draw_seq(center, agents, traj=None, other=None, heat_map=False, save=False, edge=None, path='../vis', abn_idx=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')

    shapes = []
    collide = []
    poly = agents[0].get_polygon()[0]
    shapes.append(poly)
    for i in range(1, len(agents)):
        intersect = False
        poly = agents[i].get_polygon()[0]
        for shape in shapes:
            if poly.intersects(shape):
                intersect = True
                collide.append(i)
                break
        if not intersect:
            shapes.append(poly)

    colors = list(mcolors.TABLEAU_COLORS)
    lane_color = 'black'
    alpha = 0.12
    linewidth = 3

    if heat_map:
        lane_color = 'white'
        alpha = 0.2
        linewidth = 6
    ax.axis('off')

    for j in range(center.shape[0]):
        traf_state = center[j, -1]

        x0, y0, x1, y1, = center[j, :4]

        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=1, alpha=0.2)

        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=1.5)
            # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)
    if other is not None:
        for j in range(len(other)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7, alpha=0.9)
    for i in range(len(agents)):
        if i in collide: continue

        # if i<3:
        #     color = 'red'
        #     face_color = 'black'
        # else:
        #     #break
        #     color = 'royalblue'
        #     face_color = col
        ind = i % 10
        col = colors[ind]

        # color='red'
        # face_color = 'black' #col

        traj_i = traj[:, i]
        len_t = traj_i.shape[0] - 1
        for j in range(len_t):
            # if j<3:
            #     color='red'
            #     #face_color = 'black' #col
            # else:
            #     #break
            #     color = 'red'
            x0, y0 = traj_i[j]
            x1, y1 = traj_i[j + 1]

            if abs(x0) < 60 and abs(y0) < 60 and abs(x1) < 60 and abs(y1) < 60:
                ax.plot((x0, x1), (y0, y1), '-', color=col, linewidth=1.8, marker='.', markersize=3)

        agent = agents[i]
        rect = agent.get_rect()[0]
        rect = plt.Polygon(rect, edgecolor='black',
                           facecolor=col, linewidth=0.5, zorder=10000)
        ax.add_patch(rect)

    # ax.set_facecolor('black')
    plt.autoscale()
    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)

    return plt


def draw_metrics(losses):
    fig, ax = plt.subplots(1, 3)
    x = np.arange(1, 11)
    # fig.tight_layout()

    for loss in losses:
        ax[0].plot(x, loss[0])
        ax[0].set_xlabel('iteration')
        ax[0].set_ylabel("prob loss")

        ax[1].plot(x, loss[2])
        ax[1].set_xlabel('iteration')
        ax[1].set_ylabel("velocity loss")
        # ax[1,0].plot(x, loss[2], label='vel loss')
        ax[2].plot(x, loss[3])
        ax[2].set_xlabel('iteration')
        ax[2].set_ylabel("heading loss")

    # ax.xlabel('iteration')  # X轴标签
    # ax.ylabel("loss")  # Y轴标签
    # ax.title("evaluation")  # 标题

    return plt


def draw(center, agents, other, heat_map=None, save=False, edge=None, path='../vis', abn_idx=None, vis_range=60):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.axis('equal')

    colors = list(mcolors.TABLEAU_COLORS)
    lane_color = 'black'
    alpha = 0.12
    linewidth = 8
    if heat_map:
        lane_color = 'white'
        ax.imshow(heat_map[0], extent=heat_map[1], alpha=1, origin='lower', cmap=cm.jet)
        alpha = 0.5
        linewidth = 3
        plt.xlim(heat_map[1][:2])
        plt.ylim(heat_map[1][2:])
    ax.axis('off')

    for j in range(center.shape[0]):
        traf_state = center[j, -1]

        x0, y0, x1, y1, = center[j, :4]

        if x0 == 0: break
        ax.plot((x0, x1), (y0, y1), '--', color=lane_color, linewidth=1, alpha=0.2)

        if traf_state == 1:
            color = 'red'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 2:
            color = 'yellow'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)
        elif traf_state == 3:
            color = 'green'
            ax.plot((x0, x1), (y0, y1), color=color, alpha=alpha, linewidth=linewidth, zorder=5000)

    if edge is not None:
        for j in range(len(edge)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = edge[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=1)
            # ax.arrow(x0, y0, x1-x0, y1-y0,head_width=1.5,head_length=0.75,width = 0.1)
    if other is not None:
        for j in range(len(other)):

            # if lane[j, k, -1] == 0: continue
            x0, y0, x1, y1, = other[j, :4]
            if x0 == 0: break
            ax.plot((x0, x1), (y0, y1), lane_color, linewidth=0.7)

    for i in range(len(agents)):

        ind = i % 10
        col = colors[ind]
        agent = agents[i]
        center = agent.position[0]
        if abs(center[0]) > (vis_range - 7) or abs(center[1]) > (vis_range - 7): continue
        vel = agent.velocity[0]
        rect = agent.get_rect()[0]
        rect = plt.Polygon(rect, edgecolor=lane_color,
                           facecolor=col, linewidth=0.5, zorder=10000)
        if abs(vel[0] + center[0]) < (vis_range - 2) and abs(vel[1] + center[1]) < (vis_range - 2):
            ax.plot([center[0], vel[0] + center[0]], [center[1], vel[1] + center[1]], '.-', color='lime', linewidth=1,
                    markersize=2, zorder=10000)
        ax.add_patch(rect)

    plt.autoscale()
    if save:
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)

    return plt


if __name__ == "__main__":
    loss_path = '/Users/fenglan/v2_baseline_22_08_08-05_40_27'
    with open(loss_path, 'rb+') as f:
        loss1 = pickle.load(f)
    loss_path = '/Users/fenglan/v2_traffic_ablation_22_08_09-04_56_21'
    with open(loss_path, 'rb+') as f:
        loss2 = pickle.load(f)

    losses = [loss1, loss2]

    draw_metrics(losses)
