import torch
import numpy as np
import torch
from torchmetrics import Metric
from tqdm import tqdm
import copy
import wandb

class MMD(Metric):
    full_state_update: bool = False

    def __init__(
        self,
        device,
        kernel_mul=2.0,
        kernel_num=5,
    ):
        super().__init__(full_state_update=False)

        self.add_state("mmd_sum", default=torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, device=device), dist_reduce_fx="sum")

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul**(kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def update(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        mmd_score = torch.mean(XX + YY - XY - YX)

        self.mmd_sum += mmd_score
        self.count += 1

    def compute(self):
        return self.mmd_sum / self.count

def get_metrics(self):
    self.model.eval()
    device = self.cfg['device']
    eval_data = self.eval_data_loader
    with torch.no_grad():
        mmd_metrics = {'heading': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                       'size': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                       'speed': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                       'position': MMD(device=device, kernel_mul=1.0, kernel_num=1)}
        cnt = 0
        for batch in tqdm(eval_data):

            for key in batch.keys():
                if isinstance(batch[key], torch.DoubleTensor):
                    batch[key] = batch[key].float()
                if isinstance(batch[key], torch.Tensor) and self.cfg['device'] == 'cuda':
                    batch[key] = batch[key].cuda()
            target_agent = copy.deepcopy(batch['agent'])

            output = self.model(batch, eval=True)
            pred_agent = output['agent']
            agent_num = len(pred_agent)
            pred_agent = pred_agent[1:]
            device = batch['center'].device

            source = {
                'heading': torch.tensor(normalize_angle(np.concatenate([x.heading for x in pred_agent], axis=0)),
                                        device=device),
                'size': torch.tensor(np.concatenate([x.length_width for x in pred_agent], axis=0), device=device),
                'speed': torch.tensor(np.concatenate([x.velocity for x in pred_agent], axis=0), device=device),
                'position': torch.tensor(np.concatenate([x.position for x in pred_agent], axis=0), device=device)}

            if torch.any(torch.isnan(source['speed'])):
                print('nan!')
                continue
            target = {'heading': normalize_angle(target_agent[0, 1:agent_num, [4]]),
                      'size': target_agent[0, 1:agent_num, 5:7],
                      'speed': target_agent[0, 1:agent_num, 2:4],
                      'position': target_agent[0, 1:agent_num, :2]}

            for attr, metri in mmd_metrics.items():
                # ignore empty scenes
                if agent_num <= 1:
                    continue
                if torch.all(source[attr] == 0) and torch.all(target[attr] == 0):
                    continue
                metri.update(source[attr], target[attr])

        log = {}
        for attr, metric in mmd_metrics.items():
            log[attr] = metric.compute()
        print(log)
        if not self.in_debug:
            wandb.log(log)

if __name__ == "__main__":
    heading_mmd = MMD(kernel_mul=1.0, kernel_num=1)
    size_mmd = MMD(kernel_mul=1.0, kernel_num=1)
    speed_mmd = MMD(kernel_mul=1.0, kernel_num=1)

    mmd_metrics = {
        'heading': MMD(kernel_mul=1.0, kernel_num=1),
        'size': MMD(kernel_mul=1.0, kernel_num=1),
        'speed': MMD(kernel_mul=1.0, kernel_num=1)
    }

    dims = {'heading': 2, 'size': 2, 'speed': 2}

    # iterate over all the real scenes in the dataset (or sampled dataset)
    for i in range(100):

        # number of samples for each scene
        N = np.random.randint(0, 10)

        for attr, dim in dims.items():

            # ignore empty scenes
            if N == 0:
                continue

            # obtain samples from real data
            source = torch.randn(N, dim)

            # generate samples from model
            target = torch.randn(N, dim)

            # update metric
            mmd_metrics[attr].update(source, target)

    for attr, metric in mmd_metrics.items():
        print('averaged MMD for {}: {}'.format(attr, metric.compute()))