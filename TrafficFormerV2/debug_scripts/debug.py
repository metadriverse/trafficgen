from drivingforce.TrafficTranformer.utils.arg_parse import get_parsed_args
from drivingforce.TrafficTranformer.utils.train import Trainer, get_time_str
from l5kit.configs import load_config_data

# ============================== debug =======================================
if __name__ == "__main__":
    args = get_parsed_args()
    args.distributed = False
    cfg = load_config_data('../cfg/lqy_debug_local.yaml')
    trainer = Trainer(
                      exp_name="debug_deleteme",
                      wandb_log_freq=5,
                      cfg=cfg,
                      args=args
                      )
    # trainer.load_model("F:\\hk\\drivingforce\\drivingforce\\TrafficTranformer\\TrafficTransformer_211026-120017\\saved_models\\model_11.pt")
    trainer.train()
