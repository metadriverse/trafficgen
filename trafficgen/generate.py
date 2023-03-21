from trafficgen.traffic_generator.traffic_generator import trafficgen
from trafficgen.traffic_generator.utils.utils import get_parsed_args
from trafficgen.utils.config import load_config_init

# Please keep this line here:
from trafficgen.utils.typedef import AgentType, RoadLineType, RoadEdgeType

if __name__ == "__main__":
    args = get_parsed_args()

    cfg = load_config_init(args.config)
    print('loading checkpoint...')
    trafficgen = trafficgen(cfg)
    print('Complete.\n')

    trafficgen.generate_scenarios(gif=args.gif, save_metadrive=args.save_metadrive)
