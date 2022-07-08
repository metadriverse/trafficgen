from TrafficFormerV2.utils.arg_parse import get_parsed_args
from l5kit.configs import load_config_data
from TrafficFormerV2.utils.train import Trainer
from TrafficFormerV2.load_for_metadrive import AgentType,RoadEdgeType,RoadLineType
# ============================== Main =======================================
if __name__ == "__main__":
    # ===== parse args =====
    args = get_parsed_args()
    cfg = load_config_data(f'cfg/{args.cfg}.yaml')
    #cfg['device'] = 'cuda'
    trainer = Trainer(exp_name=args.exp_name,
                      cfg=cfg,
                      args=args)
    trainer.load_model('/Users/fenglan/model_120.pt','cpu')
    #trainer.draw_generation_process()
    #trainer.generate_case_for_dynamic(10000,'/Users/fenglan/Downloads/waymo/dynamic_case')
    trainer.eval_model()
    #trainer.train()
