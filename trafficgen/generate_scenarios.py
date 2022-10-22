from trafficgen.utils.arg_parse import get_parsed_args
from trainer import Trainer
from trafficgen.utils.typedef import AgentType,RoadEdgeType,RoadLineType
import yaml
# ============================== Main =======================================


if __name__ == "__main__":

    # ===== parse args =====
    args = get_parsed_args()

    cfg_path = f'./cfg/{args.cfg}.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(exp_name=args.exp_name,
                      cfg=cfg,
                      args=args)

    device = cfg['device']

    trainer.load_model(trainer.model1,'model_weights/init',device)
    trainer.load_model(trainer.model2, 'model_weights/act', device)

    trainer.generate_scenarios()






