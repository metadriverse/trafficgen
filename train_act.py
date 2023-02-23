import pytorch_lightning as pl
from act.utils.data_utils import actDataset
#from act.utils.temp_dataset import actDataset
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
from utils.config import load_config_act
from act.model.tg_act import actuator
from utils.typedef import AgentType,RoadEdgeType,RoadLineType
import torch
# General config
from pytorch_lightning.loggers import WandbLogger
from utils.config import get_parsed_args


if __name__ == '__main__':

    args = get_parsed_args()
    cfg = load_config_act(args.config)

    if cfg['debug']:
        #wandb_logger = WandbLogger(project="trafficGen_ptl", name=args.exp_name)
        trainer = pl.Trainer(devices=1, gradient_clip_val=0.5, accelerator=cfg['device'],
                             profiler="simple")
    else:
        wandb_logger = WandbLogger(project="trafficGen_ptl",name=args.exp_name)
        trainer = pl.Trainer(max_epochs=200,logger=wandb_logger, devices=args.devices,
                             gradient_clip_val=0.5, accelerator=cfg['device'], profiler="simple", strategy=cfg['strategy'])

    train_set = actDataset(cfg)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(
                train_set, batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                shuffle=True, drop_last=False)

    val_loader = DataLoader(
        val_set, batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False, drop_last=False)

    model = actuator(cfg)
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)
