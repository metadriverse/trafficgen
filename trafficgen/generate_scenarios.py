from trafficgen.utils.arg_parse import get_parsed_args
from trainer import Trainer
import yaml
import os

# Must keep this line:
from trafficgen.utils.typedef import *

PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    args = get_parsed_args()

    # cfg_path = f'./cfg/{args.cfg}'
    cfg_path = os.path.join(PROJECT_ROOT, "trafficgen", "cfg", args.cfg)
    print("Loading config from: ", cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["data_path"] = os.path.join(PROJECT_ROOT, cfg["data_path"])
    if args.num_scenarios is not None:
        cfg["data_usage"] = args.num_scenarios

    trainer = Trainer(
        exp_name=args.exp_name,
        cfg=cfg,
        args=args
    )

    device = cfg['device']

    model_weights_folder_path = os.path.join(PROJECT_ROOT, "trafficgen", "model_weights")

    trainer.load_model(trainer.model1, os.path.join(model_weights_folder_path, 'init'), device)
    trainer.load_model(trainer.model2, os.path.join(model_weights_folder_path, 'act'), device)

    trainer.generate_scenarios(snapshot=True, gif=args.gif)





