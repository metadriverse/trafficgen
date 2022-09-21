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

    if cfg['model']=='init' or cfg['model']=='sceneGen':

        trainer.train()

        #trainer.load_model('model_weights/init_100',device)
        #trainer.get_metrics()
        #trainer.eval_init()
        #trainer.get_heatmaps()
        #trainer.get_cases()
    else:
        #trainer.load_model('model_weights/act_70', device)
        #trainer.get_metrics_for_act()
        trainer.train()
        #trainer.eval_act()
        #trainer.get_gifs()
        #trainer.get_gifs_from_gt()

