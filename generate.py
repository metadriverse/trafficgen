from traffic_generator.traffic_generator import trafficgen
from traffic_generator.utils.utils import get_parsed_args
from utils.config import load_config_init
from utils.typedef import AgentType,RoadEdgeType,RoadLineType

if __name__ == "__main__":

    args = get_parsed_args()

    cfg = load_config_init(args.config)
    print('loading checkpoint...')
    trafficgen = trafficgen(cfg)
    print('Complete.\n')

    trafficgen.generate_scenarios(gif=args.gif)
