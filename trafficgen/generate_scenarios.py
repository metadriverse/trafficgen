from utils.arg_parse import get_parsed_args
from l5kit.configs import load_config_data
from trainer import Trainer
from utils.typedef import AgentType,RoadEdgeType,RoadLineType
# ============================== Main =======================================


if __name__ == "__main__":

    # ===== parse args =====
    args = get_parsed_args()
    cfg = load_config_data(f'./cfg/{args.cfg}.yaml')
    trainer = Trainer(exp_name=args.exp_name,
                      cfg=cfg,
                      args=args)

    device = cfg['device']

    trainer.load_model(trainer.model1,'model_weights/init',device)
    trainer.load_model(trainer.model2, 'model_weights/act', device)
    trainer.place_vehicles(vis=True)
    initialized_num = cfg['init']['eval_data_usage']
    trainer.generate_traj(initialized_num,snapshot=True)




