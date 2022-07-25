import torch

def loss_v1(pred,gt):
    MSE = torch.nn.MSELoss(reduction='none')
    L1 = torch.nn.L1Loss(reduction='none')
    CLS = torch.nn.CrossEntropyLoss()

    prob_pred = pred['prob']
    velo_pred = pred['velo']
    pos_pred = pred['pos']
    heading_pred = pred['heading']
    pos_pred = pos_pred

    pos_gt = gt[:,1:, :2].unsqueeze(1).repeat(1, 6, 1, 1)
    velo_gt = gt[:,1:, 2:4].unsqueeze(1).repeat(1, 6, 1, 1)
    heading_gt = gt[:,1:, 4].unsqueeze(1).repeat(1, 6, 1)

    pred_end = pos_pred[:, :, -1]
    gt_end = pos_gt[:, :, -1]
    dist = MSE(pred_end, gt_end).mean(-1)
    min_index = torch.argmin(dist, dim=-1)

    cls_loss = CLS(prob_pred, min_index)

    pos_loss = MSE(pos_gt, pos_pred).mean(-1).mean(-1)
    pos_loss = torch.gather(pos_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

    velo_loss = MSE(velo_gt, velo_pred).mean(-1).mean(-1)
    velo_loss = torch.gather(velo_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

    heading_loss = L1(heading_gt, heading_pred).mean(-1)
    heading_loss = torch.gather(heading_loss, dim=1, index=min_index.unsqueeze(-1)).mean()

    loss_sum = 10*pos_loss+velo_loss+heading_loss+cls_loss
    loss_dict = {}
    loss_dict = {}
    loss_dict['cls_loss'] = cls_loss
    loss_dict['velo_loss'] = velo_loss
    loss_dict['heading_loss'] = heading_loss
    loss_dict['pos_loss'] = pos_loss
    return loss_sum,loss_dict