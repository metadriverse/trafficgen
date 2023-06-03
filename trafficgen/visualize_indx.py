import torch
import argparse
from trafficgen.init.utils.init_dataset import WaymoAgent
from trafficgen.traffic_generator.utils.tsne import visualize_tsne_points,visualize_tsne_images
from trafficgen.traffic_generator.utils.vis_utils import draw
import wandb

wandb.init(
    project="tsne",
)

parser = argparse.ArgumentParser()
parser.add_argument('--idx', '-i', type=int)
args = parser.parse_args()

tsne_tmp = torch.load('./tsne_tmp.pt',map_location=torch.device('cpu'))
batch_list = tsne_tmp['batch_list']

idx = args.idx
data = batch_list[idx]
agent = data['agent'][0].numpy()
agent_list = []
agent_num = agent.shape[0]
for a in range(agent_num):
    agent_list.append(WaymoAgent(agent[[a]]))
imag = draw(data['center'][0].numpy(), agent_list, other=data['rest'][0].numpy(), edge=data['bound'][0].numpy(),
     path=f'./img/{idx}.jpg', save=True)

ret = {}
ret['idx'] = wandb.Image(imag)

wandb.log(ret)

