from trafficgen.utils.arg_parse import get_parsed_args
from l5kit.configs import load_config_data
from trafficgen.utils.train import Trainer

# ============================== Main =======================================

if __name__ == "__main__":
    # ===== parse args =====
    args = get_parsed_args()
    cfg = load_config_data(f'cfg/{args.cfg}')
    trainer = Trainer(exp_name=args.exp_name,
                      cfg=cfg,
                      args=args)
    trainer.load_model('/Users/fenglan/model_120.pt', 'cpu')

    trainer.process_case_for_eval("/Users/fenglan/Downloads/waymo/v2_data", 4)

    trainer.generate_case_for_dynamic(5, '/Users/fenglan/Downloads/waymo/dynamic_case')
