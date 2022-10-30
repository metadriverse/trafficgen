import copy
import logging

import torch
import torch.nn as nn
from trafficgen.TrafficGen_act.models.critertion import loss_v1

from trafficgen.utils.model_utils import MLP_3, CG_stacked

logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0


class Actuator(nn.Module):
    """ A transformer model with wider latent space """

    def __init__(self):
        super().__init__()

        # input embedding stem
        hidden_dim = 1024
        self.CG_agent = CG_stacked(5, hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.CG_all = CG_stacked(5, hidden_dim * 2)
        self.agent_encode = MLP_3([8, 256, 512, hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)
        self.anchor_embedding = nn.Embedding(6, hidden_dim * 2)
        # self.anchor_embedding.weight.requires_grad == False
        # nn.init.orthogonal_(self.anchor_embedding.weight)
        #
        self.apply(self._init_weights)
        # prob,long_perc,lat_perc,dir(2),v_value,v_dir = 1+1+1+2+1+2
        self.pred_len = 89

        self.velo_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len * 2])
        self.pos_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len * 2])
        # self.speed_head = MLP_3([hidden_dim*2,hidden_dim,256,self.pred_len])
        self.angle_head = MLP_3([hidden_dim * 2, hidden_dim, 256, self.pred_len])

        self.prob_head = MLP_3([hidden_dim * 2, hidden_dim, 256, 1])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, data, is_training=True):

        agent = data['agent']
        agent_mask = data['agent_mask']

        all_vec = torch.cat([data['center'], data['cross'], data['bound']], dim=-2)
        line_mask = torch.cat([data['center_mask'], data['cross_mask'], data['bound_mask']], dim=1)

        polyline = all_vec[..., :4]
        polyline_type = all_vec[..., 4].to(int)
        polyline_traf = all_vec[..., 5].to(int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)

        agent_enc = self.agent_encode(agent)
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        b, a, d = agent_enc.shape

        device = agent_enc.device

        context_agent = torch.ones([b, d]).to(device)
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)

        all_context = torch.cat([context_agent, context_line], dim=-1)

        anchors = self.anchor_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
        mask = torch.ones(*anchors.shape[:-1]).to(device)
        pred_embed, _ = self.CG_all(anchors, all_context, mask)

        prob_pred = (self.prob_head(pred_embed)).squeeze(-1)
        # speed_pred = nn.ReLU()(self.speed_head(pred_embed)).view(b,6,self.pred_len)
        velo_pred = self.velo_head(pred_embed).view(b, 6, self.pred_len, 2)
        pos_pred = self.pos_head(pred_embed).view(b, 6, self.pred_len, 2).cumsum(-2)
        heading_pred = self.angle_head(pred_embed).view(b, 6, self.pred_len).cumsum(-1)

        pred = {}
        pred['prob'] = prob_pred
        pred['velo'] = velo_pred
        pred['pos'] = pos_pred
        pred['heading'] = heading_pred

        if is_training:
            loss, loss_dic = loss_v1(pred, data)
            return pred, loss, loss_dic

        else:
            # b,m,t,d = pred['pos'].shape
            # start = torch.zeros([b,m,1,2],device=device)
            # pos = torch.cat([start,pred['pos']],dim=-2).cumsum(-2)
            # pred['pos'] = pos[:,:,1:]

            return pred
