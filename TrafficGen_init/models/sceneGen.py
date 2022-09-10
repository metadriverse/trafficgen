import torch
import torch.nn as nn

from utils.model_utils import MLP_3, CG_stacked
import copy

from .init_model import initializer

copy_func = copy.deepcopy


class sceneGen(initializer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.hidden_dim = 1024
        self.feat_lstm = nn.LSTM(self.hidden_dim * 2, self.hidden_dim * 2, 2, batch_first=True)

    def _obtain_step_input(self, step_idx, data):
        # TODO Obtain the input data for the current step for feature extraction:
        # 1. process the data as if there are only agent (0, 1, ..., step_idx-1) in the scene.
        # 2. if step_idx == 0, then there is no agent in the scene.

        raise NotImplementedError

    def _obtain_step_output(self, step_idx, data):
        # TODO Obtain the output data for the current step for loss computation
        # 1. select the next agent according to the (left-right top-bottom) order
        # 2. process the data as if there is only the next agent in the scene.

        raise NotImplementedError

    def forward(self, data, random_mask=True):
        max_agent_num = torch.max(torch.sum(data['gt_distribution'], dim=1))

        all_losses = []
        all_total_loss = 0
        all_preds = []

        for step_idx in range(max_agent_num):

            # construct the input data that contain only the agents (0, 1, ..., step_idx-1)
            step_input = self._obtain_step_input(step_idx, data)
            input_feat = self.feature_extract(step_input, random_mask)

            if step_idx == 0:
                _, latent = self.feat_lstm(input_feat)
            else:
                _, latent = self.feat_lstm(input_feat, latent)

            # use hidden state of the last layer as the dist prediction feature
            K = 10
            pred_feat = latent[0][-1]
            pred_dists = self.feature_to_dists(pred_feat, K)

            # construct the output data that only contain the next agent
            step_output = self._obtain_step_output(step_idx, data)
            losses, total_loss = self.compute_loss(step_output, pred_dists)

            all_losses.append(losses)
            all_preds.append(pred_dists)
            all_total_loss += total_loss

        return all_preds, all_total_loss, all_losses
        # return pred,total_loss,losses