import pytorch_lightning as pl
from init.utils.data_utils import initDataset
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
from utils.config import load_config_init,get_parsed_args
from init.model.tg_init import initializer
import torch
from utils.typedef import AgentType,RoadEdgeType,RoadLineType
# General config
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':

    args = get_parsed_args()
    cfg = load_config_init(args.config)


    if cfg['debug']:
        trainer = pl.Trainer(devices=cfg['device_num'], gradient_clip_val=0.5, accelerator=cfg['device'],
                             profiler="simple")
    else:
        wandb_logger = WandbLogger(project="trafficGen_ptl",name=args.exp_name)
        trainer = pl.Trainer(max_epochs=100,logger=wandb_logger, devices=cfg['device_num'],
                             gradient_clip_val=0.5, accelerator=cfg['device'], profiler="simple", strategy=cfg['strategy'])

    train_set = initDataset(cfg)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size])


    train_loader = DataLoader(
                train_set, batch_size=cfg['batch_size'],
                num_workers=cfg['num_workers'],
                shuffle=True, drop_last=True)

    val_loader = DataLoader(
        val_set, batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False, drop_last=True)


    model = initializer(cfg)
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders=val_loader)

