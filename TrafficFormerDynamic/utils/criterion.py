import torch
import torch.nn as nn

class Loss(torch.nn.Module):

    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, data):
        #pred = pred['distribution']
        gt_score = data['lane_index'][0,:,1:]
        gt_vec = data['selected_vec_index'][0,:,1:]
        #gt_long = data['long_perc'][0,:,1:]
        #gt_lat = data['lat_perc'][0,:,1:]
        gt_dir = data['bucket_dir'][0,:,1:]
        gt_v_value = data['v_value'][0,:,1:]
        gt_v_dir = data['bucket_v'][0,:,1:]
        #gt_bb = data['agent'].squeeze()[:,1:,4:6]
        agent_mask = data['agent_mask'].squeeze()[:,1:]
        gt_end = torch.clone(agent_mask)
        gt_end_score = data['end_lane'][0,:,1:]
        gt_end_vec = data['end_vec'][0, :, 1:]
        # gt_end_long = data['end_long'][0, :, 1:]
        # gt_end_lat = data['end_lat'][0, :, 1:]

        #pred_long = pred['long']
        pred_score = pred['score']
        #pred_bb = pred['bounding_box']
        pred_heading = pred['heading'].squeeze()
        pred_v_value = pred["v_value"]
        pred_v_dir = pred["v_dir"]
        #pred_cat = pred["type"]
        #pred_lat = pred["lat"]
        pred_end = pred["end"]
        pred_vec = pred["vec"]

        pred_score_end = pred['score_end']
        pred_vec_end = pred['vec_end']
        # pred_long_end = pred['long_end']
        # pred_lat_end = pred['lat_end']

        #num_mask = (data['valid_num']>5).squeeze()

        CE = torch.nn.CrossEntropyLoss(reduction='none')

        BCE = torch.nn.BCELoss(reduction='mean')
        end_loss = BCE(pred_end, gt_end)
        #end_loss = torch.sum(end_loss*num_mask)/(max(1,sum(num_mask)))


        score_loss = CE(pred_score.reshape(-1, pred_score.shape[-1]), gt_score.reshape(-1).to(int))
        end_score_loss = CE(pred_score_end.reshape(-1, pred_score_end.shape[-1]), gt_end_score.reshape(-1).to(int))
        agent_mask[score_loss.reshape(*agent_mask.shape[:2])>100]=0
        agent_mask[end_score_loss.reshape(*agent_mask.shape[:2]) > 100] = 0

        #agent_mask[num_mask==0,:]=0

        score_mask = agent_mask.reshape(-1)
        agent_sum = max(1, torch.sum(agent_mask))

        score_loss = torch.sum(score_loss*score_mask)/agent_sum
        #end_score_loss = CE(pred_score_end.reshape(-1, pred_score_end.shape[-1]), gt_end_score.reshape(-1).to(int))
        end_score_loss = torch.sum(end_score_loss * score_mask) / agent_sum

        vec_loss = CE(pred_vec.reshape(-1,pred_vec.shape[-1]),gt_vec.reshape(-1).to(int))
        vec_loss = torch.sum(vec_loss*score_mask)/agent_sum
        end_vec_loss = CE(pred_vec_end.reshape(-1,pred_vec_end.shape[-1]),gt_end_vec.reshape(-1).to(int))
        end_vec_loss = torch.sum(end_vec_loss * score_mask) / agent_sum

        # long_loss = torch.square(gt_long-pred_long)
        # long_loss = torch.sum(long_loss * agent_mask) / agent_sum
        # # end_long_loss = torch.square(gt_end_long-pred_long_end)
        # # end_long_loss = torch.sum(end_long_loss * agent_mask) / agent_sum
        #
        # lat_loss = torch.square(gt_lat-pred_lat)
        # lat_loss = torch.sum(lat_loss * agent_mask) / agent_sum
        # end_lat_loss = torch.square(gt_end_lat-pred_lat_end)
        # end_lat_loss = torch.sum(end_lat_loss * agent_mask) / agent_sum


        dir_loss = CE(pred_heading.reshape(-1,pred_heading.shape[-1]),gt_dir.reshape(-1).to(int))
        dir_loss = torch.sum(dir_loss * score_mask) / agent_sum

        # bb_loss = SL1Loss(pred_bb, gt_bb).mean(-1)
        # bb_loss = torch.sum(bb_loss*agent_mask)/agent_sum

        # v_value_loss = SL1Loss(pred_v_value, gt_v_value)
        # v_value_loss = torch.sum(v_value_loss*agent_mask)/agent_sum
        v_value_loss = CE(pred_v_value.reshape(-1,pred_v_value.shape[-1]),gt_v_value.reshape(-1).to(int))
        v_value_loss = torch.sum(v_value_loss * score_mask) / agent_sum

        v_dir_loss = CE(pred_v_dir.reshape(-1,pred_v_dir.shape[-1]),gt_v_dir.reshape(-1).to(int))
        v_dir_loss = torch.sum(v_dir_loss * score_mask) / agent_sum
        #index = torch.argmax(gt_cat, axis=-1).view(-1)
        #cat_loss = CE(pred_cat.reshape(-1, pred_cat.shape[-1]), index)
        #cat_loss = torch.sum(cat_loss*score_mask)/agent_sum

        loss = score_loss+dir_loss+v_value_loss+v_dir_loss+end_loss+vec_loss+end_score_loss+end_vec_loss
        losses = {"score_loss":score_loss,
                  #"long_loss":long_loss,
                  "dir_loss":dir_loss,
                  #"bb_loss":bb_loss,
                  "v_loss":v_value_loss+v_dir_loss,
                  #"cat_loss": cat_loss,
                  #"lat_loss": lat_loss,
                  "end_loss": end_loss,
                  "vec_loss":vec_loss,
                  "end_score_loss":end_score_loss,
                  "end_vec_loss":end_vec_loss,}
                  #'end_pos_loss':end_long_loss+end_lat_loss}

        return loss,losses

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=1.0, kernel_num=1):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = 1
        return

    def guassian_kernel(self, source, target, kernel_mul=1.0, kernel_num=1, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(-1)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                  fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
