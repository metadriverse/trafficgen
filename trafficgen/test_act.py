import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from trafficgen.act.model.tg_act import act_loss
from trafficgen.act.model.tg_act import actuator
from trafficgen.act.utils.act_dataset import actDataset
from trafficgen.init.utils.init_dataset import WaymoAgent
# General config
from trafficgen.utils.config import get_parsed_args
from trafficgen.utils.config import load_config_act


def wash(batch):
    """Transform the loaded raw data to pretty pytorch tensor."""
    for key in batch.keys():
        if isinstance(batch[key], np.ndarray):
            batch[key] = Tensor(batch[key])
        if isinstance(batch[key], torch.DoubleTensor):
            batch[key] = batch[key].float()
        if 'mask' in key:
            batch[key] = batch[key].to(bool)


def get_SCR(pred):
    collide_idx = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        agent = WaymoAgent(pred[[i]])
        polygons = agent.get_polygon()
        for idx, poly in enumerate(polygons):
            for ano_idx, another_poly in enumerate(polygons):
                if poly.intersects(another_poly) and idx != ano_idx:
                    collide_idx[idx] = 1

    return np.mean(collide_idx)


if __name__ == '__main__':

    args = get_parsed_args()
    cfg = load_config_act(args.config)

    test_set = actDataset(cfg)

    dataloader = DataLoader(test_set, batch_size=4, num_workers=0, shuffle=True, drop_last=False)

    ade = []
    fde = []
    model = actuator(cfg).load_from_checkpoint('traffic_generator/ckpt/act.ckpt')
    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader)):
            pred = model(data)
            loss, loss_dict = act_loss(pred, data)
            ade.append(loss_dict['pos_loss'])
            fde.append(loss_dict['fde'])
    print(np.mean(ade))
    print(np.mean(fde))
