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
import seaborn as sns

color = sns.color_palette("colorblind")
dataset_to_color = {
    'waymo': color[2],
    'nuplan': color[0],
    'pg': color[3],
}

if __name__ == "__main__":
    tsne_tmp = {}

    wandb.init(
        project="tsne",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_parsed_args()
    cfg = load_config_init(args.config)

    test_set = InitDataset(cfg)

    data_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    model = initializer.load_from_checkpoint('traffic_generator/ckpt/init.ckpt').to(device)

    model.eval()

    datasize = len(data_loader)
    point_num = min(3000, datasize)
    vis_num = int(point_num*0.2)

    features = torch.zeros([point_num, 1024], device=device)

    ret = {}
    dataset_list = []
    batch_list = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            if idx >= point_num:
                break
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    pass
            features[idx] = model(batch)
            dataset_list.append(batch['dataset'][0])
            batch_list.append(batch)

    tsne = manifold.TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        n_jobs=-1
    )

    c_list = [dataset_to_color[c] for c in dataset_list]
    Y = tsne.fit_transform(features.cpu().numpy())

    rand_indx = list(range(point_num))
    np.random.shuffle(rand_indx)
    c_list = np.array(c_list)[rand_indx]
    Y = Y[rand_indx]

    ret['tsne_points'] = wandb.Image(visualize_tsne_points(Y,c_list,rand_indx))
    ret['tsne_points_annotated'] = wandb.Image(visualize_tsne_points(Y, c_list, rand_indx, annotated=True))
    wandb.log(ret)
    sampled_indx = rand_indx[:vis_num]
    sampled_c_list = c_list[:vis_num]

    img_path = './img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in tqdm(range(len(sampled_c_list))):
        data = batch_list[sampled_indx[i]]
        agent = data['agent'][0].cpu().numpy()
        agent_list = []
        agent_num = agent.shape[0]
        for a in range(agent_num):
            agent_list.append(WaymoAgent(agent[[a]]))
        draw(data['center'][0].cpu().numpy(), agent_list, other=data['rest'][0].cpu().numpy(), edge=data['bound'][0].cpu().numpy(), path=f'./img/{i}.jpg', save=True)

    ret = {}
    image_path = [f'./img/{i}.jpg' for i in range(vis_num)]
    ret['tsne_image'] = wandb.Image(visualize_tsne_images(Y[:vis_num, 0], Y[:vis_num, 1], image_path,sampled_c_list))
    wandb.log(ret)

    tsne_tmp['batch_list'] = batch_list
    tsne_tmp['Y'] = Y
    tsne_tmp['c_list'] = c_list
    tsne_tmp['rand_indx'] = rand_indx
    tsne_tmp['vis_num'] = vis_num
    torch.save(tsne_tmp, './tsne_tmp.pt')