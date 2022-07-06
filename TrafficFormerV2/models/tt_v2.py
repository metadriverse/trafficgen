import numpy as np
import logging
import torch
import torch.nn as nn


from TrafficFormerV2.models.utils import MLP_3,CG_stacked
import copy
logger = logging.getLogger(__name__)
copy_func = copy.deepcopy
version = 0

_Z = 1024
_H = _Z
_S = 32  # sequence length
STATE_NORM_FACTOR = 25.  # maximum typical goal distance, meters

class TransformerLWMConf(object):
    adam_eps = 1e-05
    adam_lr = 0.0003
    amp = True
    vehicle_dim = 8

    n_embd = 512
    image_size = 64
    image_channels = 3
    vecobs_size = 2
    n_action = 3
    # optimizer
    grad_clip = 200
    # transformer block params
    block_size = 31 # sequence length
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 8
    n_head = 8

class TrafficFormerV2(nn.Module):
    """ A transformer model with wider latent space """

    def __init__(self):
        super().__init__()

        # input embedding stem
        hidden_dim = 1024
        self.CG_agent = CG_stacked(5,hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.agent_encode = MLP_3([17,256,512,hidden_dim])
        self.line_encode = MLP_3([5,256,512,hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)
        #
        self.apply(self._init_weights)
        # prob,long_perc,lat_perc,dir(2),v_value,v_dir = 1+1+1+2+1+2
        self.output_head = MLP_3([hidden_dim*2,hidden_dim,256,4])

        self.prob_head = MLP_3([hidden_dim * 2, hidden_dim, 256, 1])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, data, random_mask=True):
        agent = data['line_with_agent'][...,:-2]
        agent_line_type = data['line_with_agent'][..., -2].to(int)
        agent_line_traf = data['line_with_agent'][..., -1].to(int)
        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)
        agent_mask = data['agent_mask']
        if random_mask:

            agent_mask[:,0]=1
            for i in range(agent_mask.shape[0]):
                masked_num = i%8
                agent_mask[i, 1 + masked_num:] = 0


        polyline = data['center'][...,:5]
        polyline_type = data['center'][...,5].to(int)
        polyline_traf = data['center'][..., 6].to(int)
        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)

        line_mask = data['center_mask']

        # agent features
        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        # map features features
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
        line_enc = torch.cat([line_enc,context_line],dim=-1)

        prob = self.prob_head(line_enc).squeeze(-1)
        output = self.output_head(line_enc)

        pred = torch.cat([prob.unsqueeze(-1),output],dim=-1)

        prob_gt = data['gt'][...,0]
        output_gt = data['gt'][...,1:]

        BCE = torch.nn.BCEWithLogitsLoss()

        MSE = torch.nn.MSELoss(reduction='none')
        other_loss = MSE(output,output_gt)

        prob_loss = BCE(prob,prob_gt)
        prob_loss = torch.sum(prob_loss*line_mask)/max(torch.sum(line_mask),1)

        gt_mask = prob_gt
        losses = {}
        gt_sum = torch.sum(gt_mask,dim=1).unsqueeze(-1)
        gt_sum = torch.clip(gt_sum,min=1)
        other_loss = torch.sum(other_loss*gt_mask.unsqueeze(-1),dim=1)/gt_sum

        pos_loss = other_loss[...,:2].mean()

        dir_loss = other_loss[...,2].mean()
        v_value_loss = other_loss[...,3].mean()
        #v_dir_loss = other_loss[...,4].mean()

        losses['prob_loss'] = prob_loss
        losses['pos_loss'] = pos_loss
        losses['dir_loss'] = dir_loss
        losses['v_value_loss'] = 0.01*v_value_loss
        #losses['v_dir_loss'] = v_dir_loss

        total_loss = pos_loss+prob_loss+dir_loss+0.01*v_value_loss#+v_dir_loss

        pred[...,0] = nn.Sigmoid()(pred[...,0])
        return pred,total_loss,losses
