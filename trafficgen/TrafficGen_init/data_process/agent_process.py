import numpy as np
import copy
from shapely.geometry import Polygon
import torch

def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), axis=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords

class WaymoAgent:
    def __init__(self, feature, vec_based_info=None,range=50,max_speed=30,from_inp=False):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        if from_inp:

            self.position = feature[..., :2]*self.RANGE
            self.velocity = feature[..., 2:4]*self.MAX_SPEED
            self.heading = np.arctan2(feature[..., 5],feature[..., 4])[...,np.newaxis]
            self.length_width = feature[..., 6:8]
            type = np.ones_like(self.heading)
            self.feature = np.concatenate(
            [self.position,self.velocity, self.heading,self.length_width,type],axis=-1)
            if vec_based_info is not None:
                vec_based_rep = copy.deepcopy(vec_based_info)
                vec_based_rep[..., 5:9] *= self.RANGE
                vec_based_rep[..., 2] *= self.MAX_SPEED
                self.vec_based_info = vec_based_rep

        else:
            self.feature = feature
            self.position = feature[...,:2]
            self.velocity = feature[...,2:4]
            self.heading = feature[...,[4]]
            self.length_width = feature[...,5:7]
            self.type = feature[...,[7]]
            self.vec_based_info = vec_based_info

    def get_agent(self,indx):
        return WaymoAgent(self.feature[[indx]],self.vec_based_info[[indx]])

    def get_list(self):
        bs, agent_num, feature_dim = self.feature.shape
        vec_dim = self.vec_based_info.shape[-1]
        feature = self.feature.reshape([-1,feature_dim])
        vec_rep = self.vec_based_info.reshape([-1,vec_dim])
        agent_num = feature.shape[0]
        lis = []
        for i in range(agent_num):
            lis.append(WaymoAgent(feature[[i]],vec_rep[[i]]))
        return lis

    def get_inp(self,act=False,act_inp=False):

        if act:
            return np.concatenate(
            [self.position,self.velocity, self.heading, self.length_width],axis=-1)

        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED
        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        if act_inp:
            return np.concatenate(
            [pos,velo, cos_head, sin_head,self.length_width],axis=-1)

        vec_based_rep = copy.deepcopy(self.vec_based_info)
        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED
        agent_feat = np.concatenate(
            [pos,velo, cos_head, sin_head, self.length_width,
             vec_based_rep],
            axis=-1)
        return agent_feat

    def get_rect(self,pad=0):

        l, w = (self.length_width[...,0]+pad) / 2, (self.length_width[...,1]+pad) / 2
        x1,y1 = l,w
        x2,y2 = l,-w

        point1 = rotate(x1,y1,self.heading[...,0])
        point2 = rotate(x2,y2,self.heading[...,0])
        center = self.position

        x1,y1 = point1[...,[0]],point1[...,[1]]
        x2,y2 = point2[...,[0]], point2[...,[1]]

        p1 = np.concatenate([center[...,[0]] + x1, center[...,[1]] + y1],axis=-1)
        p2 = np.concatenate([center[...,[0]] + x2, center[...,[1]] + y2],axis=-1)
        p3 = np.concatenate([center[...,[0]] - x1, center[...,[1]] - y1],axis=-1)
        p4 = np.concatenate([center[...,[0]] - x2, center[...,[1]] - y2],axis=-1)

        p1 = p1.reshape(-1, p1.shape[-1])
        p2 = p2.reshape(-1, p1.shape[-1])
        p3 = p3.reshape(-1, p1.shape[-1])
        p4 = p4.reshape(-1, p1.shape[-1])

        agent_num,dim = p1.shape

        rect_list = []
        for i in range(agent_num):
            rect = np.stack([p1[i],p2[i],p3[i],p4[i]])
            rect_list.append(rect)
        return rect_list


    def get_polygon(self):
        rect_list = self.get_rect(pad=0.25)

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a,b,c,d]))

        return poly_list