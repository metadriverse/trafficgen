import torch
import torch.nn as nn

from utils.model_utils import MLP_3, CG_stacked
import copy
from torch import Tensor
copy_func = copy.deepcopy
from utils.visual_init import get_heatmap
from TrafficGen_init.data_process.init_dataset import WaymoAgent
from random import choices
from utils.utils import get_agent_pos_from_vec
import numpy as np
class initializer(nn.Module):
    """ A transformer model with wider latent space """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # input embedding stem
        self.hidden_dim = cfg['hidden_dim']
        self.CG_agent = CG_stacked(5, self.hidden_dim)
        self.CG_line = CG_stacked(5, self.hidden_dim)
        self.agent_encode = MLP_3([17, 256, 512, self.hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, self.hidden_dim])
        self.type_embedding = nn.Embedding(20, self.hidden_dim)
        self.traf_embedding = nn.Embedding(4, self.hidden_dim)

        self.apply(self._init_weights)
        middle_layer_shape = [self.hidden_dim * 2, self.hidden_dim, 256]

        self.prob_head = MLP_3([*middle_layer_shape, 1])
        #self.pos_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.pos_head = MLP_3([*middle_layer_shape, 2])
        self.speed_head = MLP_3([*middle_layer_shape, 1])

        self.bbox_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])
        self.vel_heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])
        # self.speed_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample_from_distribution(self, pred,center_lane,repeat_num=10):
        prob = pred['prob'][0]
        max_prob=0
        for i in range(3):
            indx = choices(list(range(prob.shape[-1])), prob)[0]
            vec_logprob_ = prob[indx]
            if vec_logprob_>max_prob:
                the_indx = indx
                max_prob = max(vec_logprob_,max_prob)

        #idx_list = []
        prob_list = []
        agents_list = []
        pos = torch.clip(pred['pos'], min=-0.5, max=0.5)
        speed = torch.clip(pred['speed'], min=0)
        for i in range(repeat_num):
            # pos = torch.clip(pred['pos'].sample(),min=-0.5,max=0.5)
            # pos_logprob = pred['pos'].log_prob(pos)

            heading = torch.clip(pred['heading'].sample(),min=-np.pi/2,max=np.pi/2)
            heading_logprob = pred['heading'].log_prob(heading)

            vel_heading = torch.clip(pred['vel_heading'].sample(),min=-np.pi/2,max=np.pi/2)
            vel_heading_logprob = pred['vel_heading'].log_prob(vel_heading)

            bbox = torch.clip(pred['bbox'].sample(),min=1.5)
            bbox_logprob = pred['bbox'].log_prob(bbox)

            # speed = torch.clip(pred['speed'].sample(),min=0)
            # speed_logprob = pred['speed'].log_prob(speed)

            agents = get_agent_pos_from_vec(center_lane, pos[0], speed[0], vel_heading[0], heading[0], bbox[0])
            agents_list.append(agents)
            #pos_logprob_ = pos_logprob[0,the_indx]
            heading_logprob_ = heading_logprob[0,the_indx]
            vel_heading_logprob_ = vel_heading_logprob[0,the_indx]
            bbox_logprob_ = bbox_logprob[0,the_indx]
            #speed_logprob_ = speed_logprob[0,the_indx]
            all_prob = heading_logprob_+vel_heading_logprob_+heading_logprob_+bbox_logprob_
            prob_list.append(all_prob)

        max_indx = np.argmax(prob_list)
        max_agents = agents_list[max_indx]
        return max_agents,prob,the_indx
    def output_to_dist(self, para, n):
        # if n = 2, dim = 5 = 2 + 3, if n = 1, dim = 2 = 1 + 1

        if n == 2:
            loc, tril, diag = para[..., :2], para[..., 2], para[..., 3:]

            sigma_1 = torch.exp(diag[..., 0])
            sigma_2 = torch.exp(diag[..., 1])
            rho = torch.tanh(tril)

            cov = torch.stack([
                sigma_1 ** 2, rho * sigma_1 * sigma_2,
                rho * sigma_1 * sigma_2, sigma_2 ** 2
            ], dim=-1).view(*loc.shape[:-1], 2, 2)

            distri = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, covariance_matrix=cov)

            return distri

        if n == 1:
            loc, scale = para[..., 0], para[..., 1]
            scale = torch.exp(scale)
            distri = torch.distributions.Normal(loc, scale)

            return distri

    def feature_to_dists(self, feature, K):

        # feature dim (batch,128,2048)
        # vector distribution
        prob_pred = self.prob_head(feature).squeeze(-1)

        # position distribution： 2 dimension x y, range(-1/2,1/2)
        # pos_out dim (batch,128,10,6),
        # 10 mixtures
        # 0: mixture weight
        # 1-2: mean
        # 3-5: variance and covariance
        pos_out = self.pos_head(feature)
        speed_out = nn.ReLU()(self.speed_head(feature))

        # bbox distribution： 2 dimension length width
        bbox_out = self.bbox_head(feature).view([*feature.shape[:-1], K, -1])
        bbox_weight = bbox_out[..., 0]
        bbox_param = bbox_out[..., 1:]
        bbox_distri = self.output_to_dist(bbox_param, 2)
        bbox_weight = torch.distributions.Categorical(logits=bbox_weight)
        bbox_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(bbox_weight, bbox_distri)

        # heading distribution: 1 dimension,range(-pi/2,pi/2)
        heading_out = self.heading_head(feature).view([*feature.shape[:-1], K, -1])
        heading_weight = heading_out[..., 0]
        heading_param = heading_out[..., 1:]
        heading_distri = self.output_to_dist(heading_param, 1)
        heading_weight = torch.distributions.Categorical(logits=heading_weight)
        heading_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(heading_weight, heading_distri)

        # speed distribution: 1 dimension
        # vel heading distribution: 1 dimension,range(-pi/2,pi/2)
        vel_heading_out = self.vel_heading_head(feature).view([*feature.shape[:-1], K, -1])
        vel_heading_weight = vel_heading_out[..., 0]
        vel_heading_param = vel_heading_out[..., 1:]
        vel_heading_distri = self.output_to_dist(vel_heading_param, 1)
        vel_heading_weight = torch.distributions.Categorical(logits=vel_heading_weight)
        vel_heading_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(vel_heading_weight,
                                                                                    vel_heading_distri)
        return {'prob': prob_pred, 'pos': pos_out, 'bbox': bbox_gmm, 'heading': heading_gmm, 'speed': speed_out.squeeze(-1),
                'vel_heading': vel_heading_gmm}

    def agent_feature_extract(self, agent_feat,agent_mask, random_mask):
        agent = agent_feat[..., :-2]
        agent_line_type = agent_feat[..., -2].to(int)
        agent_line_traf = agent_feat[..., -1].to(int)
        # agent_line_traf = torch.zeros_like(agent_line_traf).to(agent.device)

        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)

        min_agent_num = self.cfg['min_agent']
        if random_mask:
            agent_mask[:, 0] = 1
            for i in range(agent_mask.shape[0]):
                masked_num = i % min_agent_num
                agent_mask[i, 1 + masked_num:] = 0

        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        b, a, d = agent_enc.shape

        context_agent = torch.ones([b, d], device=agent_feat.device)
        # agent information fusion with CG block
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        return context_agent

    def map_feature_extract(self,lane_inp,line_mask,context_agent):
        device = lane_inp.device

        polyline = lane_inp[..., :4]
        polyline_type = lane_inp[..., 4].to(int)
        polyline_traf = lane_inp[..., 5].to(int)
        # polyline_traf = torch.zeros_like(polyline_traf).to(agent.device)

        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        polyline_traf_embed = torch.zeros_like(polyline_traf_embed,device=device)

        # agent features
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        # map information fusion with CG block
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        # map context feature
        context_line = context_line.unsqueeze(1).repeat(1, line_enc.shape[1], 1)
        feature = torch.cat([line_enc, context_line], dim=-1)

        return feature

    # def feature_extract(self, data, random_mask):
    #
    #     # map features
    #     return feature

    def compute_loss(self, data, pred_dists):
        BCE = torch.nn.BCEWithLogitsLoss()
        MSE = torch.nn.MSELoss(reduction='none')
        prob_loss = BCE(pred_dists['prob'], data['gt_distribution'])

        line_mask = data['center_mask']
        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)

        gt_mask = data['gt_distribution']
        gt_sum = torch.clip(torch.sum(gt_mask, dim=1).unsqueeze(-1), min=1)
        # long lat loss

        # pos_loss = -pred_dists['pos'].log_prob(data['gt_long_lat'])
        # pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()
        pos_loss = MSE(pred_dists['pos'],data['gt_long_lat']).mean(-1)
        pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()

        speed_loss = MSE(pred_dists['speed'],data['gt_speed'])
        speed_loss = (torch.sum(speed_loss * gt_mask, dim=1) / gt_sum).mean()

        # speed_loss = -pred_dists['speed'].log_prob(data['gt_speed'])
        # speed_loss = (torch.sum(speed_loss * gt_mask, dim=1) / gt_sum).mean()

        bbox_loss = -pred_dists['bbox'].log_prob(data['gt_bbox'])
        bbox_loss = (torch.sum(bbox_loss * gt_mask, dim=1) / gt_sum).mean()


        vel_heading_loss = -pred_dists['vel_heading'].log_prob(data['gt_vel_heading'])
        vel_heading_loss = (torch.sum(vel_heading_loss * gt_mask, dim=1) / gt_sum).mean()

        heading_loss = -pred_dists['heading'].log_prob(data['gt_heading'])
        heading_loss = (torch.sum(heading_loss * gt_mask, dim=1) / gt_sum).mean()

        losses = {}

        losses['prob_loss'] = prob_loss
        losses['pos_loss'] = pos_loss
        losses['heading_loss'] = heading_loss
        losses['speed_loss'] = speed_loss
        losses['vel_heading_loss'] = vel_heading_loss
        losses['bbox_loss'] = bbox_loss

        total_loss = prob_loss + pos_loss + heading_loss + speed_loss + vel_heading_loss + bbox_loss

        return losses, total_loss

    def inference(self, data):

        agent_num = data['agent_mask'].sum().item()

        idx_list = []

        vec_indx = data['vec_based_rep'][...,0]
        idx_list.append(vec_indx[0,0].item())
        shapes = []

        ego_agent = data['agent'][0,[0]].cpu().numpy()
        ego_agent = WaymoAgent(ego_agent)
        ego_poly = ego_agent.get_polygon()[0]

        shapes.append(ego_poly)

        minimum_agent = self.cfg['pad_num']
        center = data['center'][0]
        center_mask = data['center_mask'][0].cpu().numpy()
        pred_list = []
        pred_list.append(ego_agent)
        heat_maps = []
        prob_list = []

        for i in range(1,max(agent_num,minimum_agent)):
            data['agent_mask'][:, :i] = 1
            data['agent_mask'][:,i:]=0

            pred, _, _ = self.forward(data, False, False)

            pred['prob'][:,idx_list]=0
            cnt=0
            while cnt<10:
                agents, prob, indx = self.sample_from_distribution(pred,center)
                the_agent = agents.get_agent(indx)

                poly = the_agent.get_polygon()[0]

                intersect = False
                for shape in shapes:
                    if poly.intersects(shape):
                        intersect = True
                        break
                if not intersect:
                    shapes.append(poly)
                    break
                else:
                    cnt+=1
                    continue
            pred_list.append(the_agent)
            data['agent_feat'][:,i] = Tensor(the_agent.get_inp())
            idx_list.append(indx)

            heat_maps.append(get_heatmap(agents.position[:,0][center_mask], agents.position[:, 1][center_mask],prob[center_mask].cpu().numpy(), 20))
            prob_list.append(prob)

        output = {}
        output['agent'] = pred_list
        output['prob'] = prob_list
        output['heat_maps'] = heat_maps

        return output
    def forward(self, data, random_mask=True, eval=False):
        if eval==True:
            return self.inference(data)

        context_agent = self.agent_feature_extract(data['agent_feat'],data['agent_mask'],random_mask)

        feature = self.map_feature_extract(data['lane_inp'],data['lane_mask'],context_agent)
        center_num = data['center'].shape[1]
        feature = feature[:,:center_num]
        # Sample location, bounding box, heading and velocity.
        K = 10

        pred_dists = self.feature_to_dists(feature, K)
        losses, total_loss = self.compute_loss(data, pred_dists)

        pred = pred_dists
        pred['prob'] = nn.Sigmoid()(pred['prob'])


        return pred, total_loss, losses
