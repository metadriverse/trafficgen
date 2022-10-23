from trafficgen.utils.arg_parse import get_parsed_args
from trainer import Trainer
import yaml
import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":
    args = get_parsed_args()

    # cfg_path = f'./cfg/{args.cfg}'
    cfg_path = os.path.join(PROJECT_ROOT, "trafficgen", "cfg", args.cfg)
    print("Loading config from: ", cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg["data_path"] = os.path.join(PROJECT_ROOT, cfg["data_path"])

    trainer = Trainer(
        exp_name=args.exp_name,
        cfg=cfg,
        args=args
    )

    device = cfg['device']

    trainer.load_model(trainer.model1, 'model_weights/init', device)
    trainer.load_model(trainer.model2, 'model_weights/act', device)

    trainer.generate_scenarios()





