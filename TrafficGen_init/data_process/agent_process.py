import numpy as np
import copy
from shapely.geometry import Polygon


class WaymoAgent:
    def __init__(self, feature, vec_based_info=None,range=50,max_speed=30):
        # index of xy,v,lw,yaw,type,valid

        self.RANGE = range
        self.MAX_SPEED = max_speed

        #self.bs,self.agent_num,_ = feature.shape
        self.feature = feature

        self.position = feature[...,:2]
        self.velocity = feature[...,2:4]
        self.heading = feature[...,[4]]
        self.length_width = feature[...,5:7]
        self.type = feature[...,[7]]
        self.vec_based_info = vec_based_info

    def get_agent(self,indx):
        return WaymoAgent(self.feature[[indx]],self.vec_based_info[[indx]])


    def get_inp(self):
        pos = self.position / self.RANGE
        velo = self.velocity / self.MAX_SPEED

        cos_head = np.cos(self.heading)
        sin_head = np.sin(self.heading)

        vec_based_rep = copy.deepcopy(self.vec_based_info)

        vec_based_rep[..., 5:9] /= self.RANGE
        vec_based_rep[..., 2] /= self.MAX_SPEED

        agent_feat = np.concatenate(
            [pos,velo, cos_head, sin_head, self.length_width,
             vec_based_rep],
            axis=-1)
        return agent_feat

    def get_rect(self):

        l, w = self.length_width[...,[0]] / 2, self.length_width[...,[1]] / 2
        yaw = self.heading
        center = self.position
        theta = np.arctan(w / l)
        theta[np.where(np.isnan(theta))]=0

        s1 = np.sqrt(l ** 2 + w ** 2)
        x1 = abs(np.cos(theta + yaw) * s1)
        y1 = abs(np.sin(theta + yaw) * s1)
        x2 = abs(np.cos(theta - yaw) * s1)
        y2 = abs(np.sin(theta - yaw) * s1)

        p1 = np.concatenate([center[...,[0]] + x1, center[...,[1]] + y1],axis=-1)
        p2 = np.concatenate([center[...,[0]] + x2, center[...,[1]] - y2],axis=-1)
        p3 = np.concatenate([center[...,[0]] - x1, center[...,[1]] - y1],axis=-1)
        p4 = np.concatenate([center[...,[0]] - x2, center[...,[1]] + y2],axis=-1)

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
        rect_list = self.get_rect()

        poly_list = []
        for i in range(len(rect_list)):
            a = rect_list[i][0]
            b = rect_list[i][1]
            c = rect_list[i][2]
            d = rect_list[i][3]
            poly_list.append(Polygon([a,b,c,d]))

        return poly_list