import torch
import torch.nn as nn

from utils.model_utils import MLP_3, CG_stacked
import copy

from .init_model import initializer
from random import choices
copy_func = copy.deepcopy
from utils.utils import get_agent_pos_from_vec
import numpy as np
class sceneGen(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # input embedding stem
        self.hidden_dim = cfg['hidden_dim']
        self.feat_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, 2, batch_first=True)
        self.CG_line = CG_stacked(5, self.hidden_dim)
        self.agent_encode = MLP_3([17, 256, 512, self.hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, self.hidden_dim])
        self.type_embedding = nn.Embedding(20, self.hidden_dim)
        self.traf_embedding = nn.Embedding(4, self.hidden_dim)

        self.apply(self._init_weights)
        middle_layer_shape = [self.hidden_dim * 2, self.hidden_dim, 256]

        self.prob_head = MLP_3([*middle_layer_shape, 1])
        # self.pos_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.pos_head = MLP_3([*middle_layer_shape, 2])
        self.speed_head = MLP_3([*middle_layer_shape, 1])

        self.bbox_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])
        self.vel_heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
        return {'prob': prob_pred, 'pos': pos_out, 'bbox': bbox_gmm, 'heading': heading_gmm,
                'speed': speed_out.squeeze(-1),
                'vel_heading': vel_heading_gmm}
    def _obtain_step_input(self, step_idx, data):
        # TODO Obtain the input data for the current step for feature extraction:
        # 1. process the data as if there are only agent (0, 1, ..., step_idx-1) in the scene.
        # 2. if step_idx == 0, then there is no agent in the scene.
        data['agent_mask'][:,:step_idx] = True
        data['agent_mask'][:, step_idx:] = False
        return
    def compute_loss(self, data, pred_dists,mask,agent_vec_idx,step_idx):
        BCE = torch.nn.BCEWithLogitsLoss()
        MSE = torch.nn.MSELoss(reduction='none')
        prob_loss = BCE(pred_dists['prob'], data['gt_distri'][:,step_idx])

        line_mask = data['center_mask']
        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)
        # long lat loss
        mask = mask.unsqueeze(-1)

        gather2 = agent_vec_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)
        gather1 = agent_vec_idx.unsqueeze(-1)
        gt_pos = torch.gather(data['gt_long_lat'],1,gather2)
        gt_bbox = torch.gather(data['gt_bbox'],1,gather2)
        gt_speed = torch.gather(data['gt_speed'],1,gather1)
        gt_vel_heading = torch.gather(data['gt_vel_heading'], 1, gather1)
        gt_heading = torch.gather(data['gt_heading'], 1, gather1)

        pos_loss = MSE(pred_dists['pos'],gt_pos).mean(-1)
        pos_loss = (pos_loss*mask).sum()/mask.sum()

        speed_loss = MSE(pred_dists['speed'],gt_speed)
        speed_loss = (speed_loss*mask).sum()/mask.sum()

        bbox_loss = -pred_dists['bbox'].log_prob(gt_bbox)
        bbox_loss = (bbox_loss*mask).sum()/mask.sum()


        vel_heading_loss = -pred_dists['vel_heading'].log_prob(gt_vel_heading)
        vel_heading_loss = (vel_heading_loss*mask).sum()/mask.sum()

        heading_loss = -pred_dists['heading'].log_prob(gt_heading)
        heading_loss = (heading_loss*mask).sum()/mask.sum()

        losses = {}

        losses['prob_loss'] = prob_loss
        losses['pos_loss'] = pos_loss
        losses['heading_loss'] = heading_loss
        losses['speed_loss'] = speed_loss
        losses['vel_heading_loss'] = vel_heading_loss
        losses['bbox_loss'] = bbox_loss

        total_loss = prob_loss + pos_loss + heading_loss + speed_loss + vel_heading_loss + bbox_loss

        return losses, total_loss
    def sample_from_distribution(self, pred,center_lane,repeat_num=10):


        prob_list = []
        agents_list = []
        pos = torch.clip(pred['pos'], min=-0.5, max=0.5)
        speed = torch.clip(pred['speed'], min=0)
        for i in range(repeat_num):

            heading = torch.clip(pred['heading'].sample(),min=-np.pi/2,max=np.pi/2)
            heading_logprob = pred['heading'].log_prob(heading)

            vel_heading = torch.clip(pred['vel_heading'].sample(),min=-np.pi/2,max=np.pi/2)
            vel_heading_logprob = pred['vel_heading'].log_prob(vel_heading)

            bbox = torch.clip(pred['bbox'].sample(),min=1.5)
            bbox_logprob = pred['bbox'].log_prob(bbox)

            agents = get_agent_pos_from_vec(center_lane, pos[0], speed[0], vel_heading[0], heading[0], bbox[0])
            agents_list.append(agents)
            #pos_logprob_ = pos_logprob[0,the_indx]
            heading_logprob_ = heading_logprob[0,0]
            vel_heading_logprob_ = vel_heading_logprob[0,0]
            bbox_logprob_ = bbox_logprob[0,0]
            #speed_logprob_ = speed_logprob[0,the_indx]
            all_prob = heading_logprob_+vel_heading_logprob_+bbox_logprob_
            prob_list.append(all_prob)

        max_indx = np.argmax(prob_list)
        max_agents = agents_list[max_indx]
        return max_agents

    def agent_feature_extract(self, agent_feat):
        agent = agent_feat[..., :-2]
        agent_line_type = agent_feat[..., -2].to(int)
        agent_line_traf = agent_feat[..., -1].to(int)
        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)
        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        return agent_enc

    def inference(self, data):
        max_agent_num = torch.max(torch.sum(data['gt_distribution'], dim=1)).to(int).item()

        all_preds = []
        bs,lane_num = data['gt_distribution'].shape

        for step_idx in range(1, max_agent_num):
            agent_vec_indx = data['agent_vec_indx'][:, step_idx]
            for i in range(bs):
                data['gt_distri'][i, step_idx, agent_vec_indx[i]] = 1
        generated_indx = []
        generated_indx.append(data['agent_vec_indx'][0,0].item())
        for step_idx in range(1, max_agent_num):
            # construct the input data that contain only the agents (0, 1, ..., step_idx-1)
            agent_enc = self.agent_feature_extract(data['agent_feat'])

            input_feat = agent_enc[:, :step_idx]
            if step_idx == 1:
                _, latent = self.feat_lstm(input_feat)
            else:
                _, latent = self.feat_lstm(input_feat, latent)

            K = 10
            pred_feat = latent[0][-1]

            feature = self.map_feature_extract(data['lane_inp'],data['lane_mask'],pred_feat)
            center_num = data['center'].shape[1]
            feature = feature[:, :center_num]

            prob_pred = self.prob_head(feature).squeeze(-1)

            prob = nn.Sigmoid()(prob_pred[0])
            prob = prob*data['center_mask'][0]
            prob[generated_indx]=0
            max_prob = -10
            for i in range(3):
                indx = choices(list(range(prob.shape[-1])), prob)[0]
                vec_logprob_ = prob[indx]
                if vec_logprob_ > max_prob:
                    the_indx = indx
                    max_prob = max(vec_logprob_, max_prob)
            generated_indx.append(the_indx)

            feature = feature[:,[the_indx]]

            pred_dists = self.feature_to_dists(feature, K)

            agent = self.sample_from_distribution(pred_dists,data['center'][0,[the_indx]])

            next_inp = agent.get_inp()
            next_inp = torch.tensor(next_inp,device=data['agent_feat'].device)
            data['agent_feat'][0,step_idx] = next_inp
            all_preds.append(agent)

        output = {}
        output['agent'] = all_preds
        return output

    def forward(self, data, eval=False):
        if eval==True:
            return self.inference(data)

        max_agent_num = torch.max(torch.sum(data['gt_distribution'], dim=1)).to(int).item()
        max_agent_num = min(max_agent_num,16)
        all_losses = []
        all_total_loss = 0
        all_preds = []
        data['agent_mask_gt'] = copy.deepcopy(data['agent_mask'])
        bs,lane_num = data['gt_distribution'].shape
        device =  data['gt_distribution'].device
        gt_distri = torch.zeros([bs,max_agent_num,lane_num],device=device)
        data['gt_distri'] = gt_distri
        agent_enc = self.agent_feature_extract(data['agent_feat'])

        for step_idx in range(1, max_agent_num):
            agent_vec_indx = data['agent_vec_indx'][:, step_idx]
            for i in range(bs):
                data['gt_distri'][i, step_idx, agent_vec_indx[i]] = 1

        for step_idx in range(1, max_agent_num):
            # construct the input data that contain only the agents (0, 1, ..., step_idx-1)
            input_feat = agent_enc[:,:step_idx]

            if step_idx == 1:
                _, latent = self.feat_lstm(input_feat)
            else:
                _, latent = self.feat_lstm(input_feat, latent)

            # use hidden state of the last layer as the dist prediction feature
            K = 10
            pred_feat = latent[0][-1]

            feature = self.map_feature_extract(data['lane_inp'],data['lane_mask'],pred_feat)
            center_num = data['center'].shape[1]
            feature = feature[:, :center_num]

            prob_pred = self.prob_head(feature).squeeze(-1)
            bs,lane_num,feature_dim = feature.shape

            agent_vec_indx = data['agent_vec_indx'][:, step_idx]
            gather_feat = agent_vec_indx.view(bs,1,1).repeat(1,1,feature_dim)
            feature = torch.gather(feature,1,gather_feat)

            pred_dists = self.feature_to_dists(feature, K)
            pred_dists['prob'] = prob_pred

            mask = data['agent_mask_gt'][:, step_idx]
            losses, total_loss = self.compute_loss(data, pred_dists,mask,agent_vec_indx,step_idx)
            all_losses.append(losses)
            all_preds.append(pred_dists)
            all_total_loss += total_loss

        loss_num = max_agent_num-1
        keys = all_losses[0].keys()
        all_ = {}
        for key in keys:
            all_[key]=0
        for i in range(loss_num):
            for key in keys:
                all_[key]+=all_losses[i][key]/loss_num
        all_total_loss/=(max_agent_num-1)

        return all_preds, all_total_loss, all_

