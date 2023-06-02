import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn import manifold
from trafficgen.init.utils.init_dataset import WaymoAgent
from trafficgen.traffic_generator.utils.tsne import visualize_tsne_points,visualize_tsne_images
import wandb
from trafficgen.init.model.tg_feature import initializer
from trafficgen.init.utils.init_dataset import InitDataset
from trafficgen.utils.config import load_config_init, get_parsed_args
from trafficgen.utils.utils import setup_seed,wash
from torch.utils.data import DataLoader
from trafficgen.traffic_generator.utils.vis_utils import draw

dataset_to_color = {
    'waymo': 'red',
    'nuplan': 'blue',
    'pg': 'green',
}

if __name__ == "__main__":
    wandb.init(
        project="tsne",
    )
    vis_num = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_parsed_args()
    cfg = load_config_init(args.config)

    test_set = InitDataset(cfg)

    data_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    model = initializer.load_from_checkpoint('traffic_generator/ckpt/init_sn.ckpt').to(device)

    model.eval()

    datasize = len(data_loader)
    features = torch.zeros([datasize, 1024], device=device)

    ret = {}
    dataset_list = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):

            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    pass
            features[idx] = model(batch)
            dataset_list.append(batch['dataset'][0])

    tsne = manifold.TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        n_jobs=-1
    )

    c_list = [dataset_to_color[c] for c in dataset_list]
    Y = tsne.fit_transform(features.cpu().numpy())

    rand_indx = list(range(datasize))
    np.random.shuffle(rand_indx)
    c_list = np.array(c_list)[rand_indx]
    Y = Y[rand_indx]

    ret['tsne_points'] = wandb.Image(visualize_tsne_points(Y,c_list))

    sampled_indx = rand_indx[:vis_num]
    c_list = c_list[sampled_indx]

    img_path = './img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in range(vis_num):
        data = data_loader.dataset[sampled_indx[i]]
        agent = data['agent']
        agent_list = []
        agent_num = agent.shape[0]
        for a in range(agent_num):
            agent_list.append(WaymoAgent(agent[[a]]))
        draw(data['center'], agent_list, other=data['rest'], edge=data['bound'], path=f'./img/{i}.jpg', save=True)

    image_path = [f'./img/{i}.jpg' for i in range(vis_num)]
    ret['tsne_image'] = wandb.Image(visualize_tsne_images(Y[:vis_num, 0], Y[:vis_num, 1], image_path,c_list))
    wandb.log(ret)