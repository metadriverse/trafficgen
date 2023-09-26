import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn import manifold
from trafficgen.init.utils.init_dataset import WaymoAgent
from trafficgen.traffic_generator.utils.tsne import visualize_tsne_points
from trafficgen.init.model.tg_feature import initializer
from trafficgen.init.utils.init_dataset import InitDataset
from trafficgen.utils.config import load_config_init, get_parsed_args
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_parsed_args()
    cfg = load_config_init(args.config)

    test_set = InitDataset(cfg)

    data_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=True, drop_last=True)

    model = initializer.load_from_checkpoint('traffic_generator/ckpt/init.ckpt').to(device)

    model.eval()

    datasize = len(data_loader)
    point_num = min(3000, datasize)
    vis_num = int(point_num*0.1)

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
            batch_to_save = {}
            for k in batch.keys():
                if 'center' == k or 'agent' == k or 'bound' == k or 'rest' == k:
                    batch_to_save[k] = batch[k]
                else:
                    continue
            batch_list.append(batch_to_save)

    # Draw a 3D TSNE
    tsne = manifold.TSNE(
        n_components=3,
        init='pca',
        learning_rate='auto',
        n_jobs=-1
    )

    c_list = [dataset_to_color[c] for c in dataset_list]
    Y = tsne.fit_transform(features.cpu().numpy())

    np.savetxt("embeddings.tsv", Y, delimiter="\t")
    with open("metadata.tsv", "w") as f:
        for label in dataset_list:
            f.write(f"{label}\n")

    rand_indx = list(range(point_num))
    np.random.shuffle(rand_indx)
    c_list = np.array(c_list)[rand_indx]
    Y = Y[rand_indx]

    plt = visualize_tsne_points(Y,c_list,rand_indx)
    plt.show()

    sampled_indx = rand_indx[:vis_num]
    sampled_c_list = c_list[:vis_num]

    img_path = './img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    for i in tqdm(range(len(c_list))):
        data = batch_list[i]
        agent = data['agent'][0].cpu().numpy()
        agent_list = []
        agent_num = agent.shape[0]
        for a in range(agent_num):
            agent_list.append(WaymoAgent(agent[[a]]))
        draw(data['center'][0].cpu().numpy(), agent_list, other=data['rest'][0].cpu().numpy(), edge=data['bound'][0].cpu().numpy(), path=f'./img/{i}.jpg', save=True)

    ret = {}
    # Save visualization of each scanario, will be used to generate sprite images in tensorboard example
    image_path = [f'./img/{i}.jpg' for i in range(vis_num)]

    tsne_tmp['batch_list'] = batch_list
    tsne_tmp['Y'] = Y
    tsne_tmp['c_list'] = c_list
    tsne_tmp['rand_indx'] = rand_indx
    tsne_tmp['vis_num'] = vis_num
    torch.save(tsne_tmp, './tsne_tmp.pt')