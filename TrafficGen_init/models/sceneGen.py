import torch
import torch.nn as nn

from utils.model_utils import MLP_3,CG_stacked
import copy

copy_func = copy.deepcopy

class initializer(nn.Module):


    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        # input embedding stem
        hidden_dim = 1024
        self.CG_agent = CG_stacked(5,hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.agent_encode = MLP_3([17,256,512,hidden_dim])
        self.line_encode = MLP_3([4,256,512,hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)

    def forward(self, data):
        agent = data['agent_feat'][...,:-2]

        agent_line_type = data['agent_feat'][..., -2].to(int)
        agent_line_traf = data['agent_feat'][..., -1].to(int)
        #agent_line_traf = torch.zeros_like(agent_line_traf).to(agent.device)

        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)
        agent_mask = data['agent_mask']

        polyline = data['lane_inp'][...,:4]
        polyline_type = data['lane_inp'][...,4].to(int)
        polyline_traf = data['lane_inp'][..., 5].to(int)


        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        polyline_traf_embed = torch.zeros_like(polyline_traf_embed).to(agent.device)

        line_mask = data['center_mask']

        # agent features
        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        # map features
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed

        b,a,d = agent_enc.shape

        device = agent_enc.device

        context_agent = torch.ones([b,d]).to(device)
        # agent information fusion with CG block
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent,agent_mask)

        # map information fusion with CG block
        line_enc, context_line= self.CG_line(line_enc, context_agent, line_mask)
        # map context feature
        context_line = context_line.unsqueeze(1).repeat(1,line_enc.shape[1],1)

        # this is the vector feature for each of 128 lane vectors, each lane vector is a 5m long road center lane
        # this feature equals to the grid feature of a image. For example, in sceneGen the image is splited as 0.25m grid.
        feature = torch.cat([line_enc,context_line],dim=-1)



        return


