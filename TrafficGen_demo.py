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

    #trainer.load_model('model_weights/act_20.pt','cpu')
    #trainer.draw_generation_process(vis=False,save_path="TrafficGen_act/data_sample/eval")
    #trainer.eval_model()
    trainer.train()

    #trainer.draw_showcase()
    #trainer.draw_gifs(vis=False,save_path='/Users/fenglan/Downloads/waymo/generated_metadrive')
    #trainer.generate_case_for_metadrive('/Users/fenglan/select')

