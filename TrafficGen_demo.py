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

    if cfg['model']=='init':
        trainer.load_model('model_weights/init_120.pt',device)
        trainer.eval_init()
        #trainer.draw_generation_process(vis=False,save=True)
    else:
        trainer.load_model('model_weights/act_80.pt', device)
        trainer.draw_gifs(vis=True)

