import copy

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trafficgen.init.model.tg_init import initializer
from trafficgen.init.utils.init_dataset import initDataset
from trafficgen.utils.config import load_config_init, get_parsed_args
from trafficgen.utils.evaluation import MMD
from trafficgen.utils.utils import normalize_angle, setup_seed


def wash(batch):
    """Transform the loaded raw data to pretty pytorch tensor."""
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray):
            batch[key] = Tensor(batch[key])
        if isinstance(batch[key], torch.DoubleTensor):
            batch[key] = batch[key].float()
        if 'mask' in key:
            batch[key] = batch[key].to(bool)

if __name__ == '__main__':

    setup_seed(42)
    args = get_parsed_args()
    cfg = load_config_init(args.config)

    test_set = initDataset(cfg)

    data_loader = DataLoader(
        test_set, batch_size=1, num_workers=0, shuffle=False, drop_last=True
    )

    model = initializer.load_from_checkpoint('traffic_generator/ckpt/init.ckpt')

    model.eval()
    context_num = 1
    data_path = cfg['data_path']
    device = cfg['device']

    mmd_metrics = {'heading': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                   'size': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                   'speed': MMD(device=device, kernel_mul=1.0, kernel_num=1),
                   'position': MMD(device=device, kernel_mul=1.0, kernel_num=1)}

    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader)):
            batch = copy.deepcopy(data)
            wash(batch)

            # Call the initialization model
            # The output is a dict with these keys: center, rest, bound, agent
            output = model.inference(batch, context_num=context_num)

            pred_agent = output['agent']
            agent_num = len(pred_agent)
            if agent_num==1:
                continue
            target_agent = batch['agent']
            pred_agent = pred_agent[1:]
            source = {
                'heading': torch.tensor(normalize_angle(np.concatenate([x.heading for x in pred_agent], axis=0)),
                                        device=device),
                'size': torch.tensor(np.concatenate([x.length_width for x in pred_agent], axis=0), device=device),
                'speed': torch.tensor(np.concatenate([x.velocity for x in pred_agent], axis=0), device=device),
                'position': torch.tensor(np.concatenate([x.position for x in pred_agent], axis=0), device=device)}

            target = {'heading': normalize_angle(target_agent[0, 1:agent_num, [4]]),
                      'size': target_agent[0, 1:agent_num, 5:7],
                      'speed': target_agent[0, 1:agent_num, 2:4],
                      'position': target_agent[0, 1:agent_num, :2]}
            #target = source
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

