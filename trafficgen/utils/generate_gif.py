import imageio
import os
from tqdm import tqdm
import numpy as np
#home_path = '../TrafficFormerV2/heatmap/selected'
home_path = '/Users/fenglan/Desktop/CUHK/TrafficGen/vis/gif'
home_path = '/Users/fenglan/Desktop/CUHK/TrafficGen/cases/simulation'
#home_path = '/Users/fenglan/Desktop/CUHK/TrafficGen/vis/heatmap'
all_file = os.listdir(home_path)
number_list = []
for x in all_file:
    try:
        number_list.append(int(x))
    except:
        pass
number_list = np.sort(number_list)
# for i in number_list:
for i in range(10,13):
    path = os.path.join(home_path,f'{i}')
    images = []
    file_list = os.listdir(path)
    gif_list = []
    for x in file_list:
        if not 'DS' in x:
            gif_list.append(x)
    gif_list = np.sort(gif_list)
    #for j in tqdm(gif_list):
    for j in tqdm(range(40)):
        file_name = os.path.join(path,f'{j}.png')
        img = imageio.imread(file_name)

        # if j==1:
        #     h_,w_ = img.shape[0]/2,img.shape[1]/2
        # else:
        #     h,w = img.shape[0],img.shape[1]
        #     centerx,centery = int(h/2),int(w/2)
        #     start_x,endx = int(centerx-h_),int(centerx+h_)
        #     start_y,endy = int(centery-w_),int(centery+w_)
        #    img = img[start_x:endx,start_y:endy]
        images.append(img)
    output = os.path.join(home_path,'gif_processed',f'movie_{i}.gif')
    imageio.mimsave(output, images,duration=0.15)