import pickle
import numpy as np
import sys
import os
from tqdm import tqdm
#from trafficgen.utils.typedef import AgentType, RoadEdgeType, RoadLineType

ALL_TYPE = {
    'LANE_FREEWAY': 1,
    'LANE_SURFACE_STREET': 2,
    'LANE_BIKE_LANE': 3,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 6,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 7,
    'ROAD_LINE_SOLID_DOUBLE_WHITE': 8,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW': 9,
    'ROAD_LINE_BROKEN_DOUBLE_YELLOW': 10,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 11,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW': 12,
    'ROAD_LINE_PASSING_DOUBLE_YELLOW': 13,
    'ROAD_EDGE_BOUNDARY': 15,
    'ROAD_EDGE_MEDIAN': 16,
    'STOP_SIGN': 17,
    'CROSS_WALK': 18,
    'SPEED_BUMP': 19,
}


SAMPLE_NUM = 10

def down_sampling(line):
    # if is center lane
    point_num = line.shape[0]

    ret = []

    if point_num < SAMPLE_NUM:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0, point_num, SAMPLE_NUM):
            ret.append(line[i])

    return ret

def MDdata_to_initdata(MDdata):
    ret = {}

    ret['id'] = MDdata['id']

    tracks = MDdata['tracks']
    tls = MDdata['dynamic_map_states']
    map_feat = MDdata['map_features']
    sdc_idx = MDdata['sdc_track_index']

    track_len = 190
    all_agent = np.zeros([track_len,len(tracks),9],dtype='float32')

    for indx,(id, track) in enumerate(tracks.items()):
        if track['type'] != 'VEHICLE':continue
        if id == sdc_idx: sdc_idx = indx
        all_agent[:, indx, :2] = track['position'][:track_len, :2]
        all_agent[:, indx, 2:4] = track['velocity'][:track_len]
        all_agent[:, indx, 4] = track['heading'][:track_len, 0]
        all_agent[:, indx, 5:7] = track['size'][:track_len, :2]
        all_agent[:, indx, 7] = 1
        all_agent[:, indx, 8] = track['valid'][:track_len, 0]

    sdc = all_agent[:,sdc_idx].copy()
    all_agent[:,sdc_idx] = all_agent[:,0]
    all_agent[:,0] = sdc
    ret['all_agent'] = all_agent

    traffic_light = []
    for i in range(track_len):
        t_l_t = []
        tls_t = tls[i]
        for j in range(len(tls_t)):
            t_l_t_j = np.zeros(6,dtype='float32')
            t_l_t_j[0] = tls_t[j]['lane']
            t_l_t_j[1:3] = tls_t[j]['stop_point'][:2]
            state = tls_t[j]['state']
            if 'GO' in state:
                t_l_t_j[4] = 3
            elif 'CAUTION' in state:
                t_l_t_j[4] = 2
            elif 'STOP' in state:
                t_l_t_j[4] = 1
            else:
                t_l_t_j[4] = 0
            t_l_t_j[5] = 1 if t_l_t_j[4] else 0
            t_l_t.append(t_l_t_j)
        traffic_light.append(t_l_t)

    ret['traffic_light']=traffic_light


    lanes = []
    for indx,(id, lane) in enumerate(map_feat.items()):
        try:
            lane_num = lane['polyline'].shape[0]
        except:
            lane['polyline'] = lane['position'][np.newaxis]
            lane_num = lane['polyline'].shape[0]
        poly_unsampled = lane['polyline'][:,:2]

        poly = down_sampling(poly_unsampled)

        a_lane = np.zeros([len(poly), 4],dtype='float32')


        a_lane[:,:2] = np.array(poly)
        a_lane[:,2] = ALL_TYPE[lane['type']]
        a_lane[:,3] = 1
        lanes.append(a_lane)
    lanes = np.concatenate(lanes,axis=0)

    ret['lane'] = lanes

    return ret

if __name__ == '__main__':
    raw_data_path = sys.argv[1]
    data_num = int(sys.argv[2])
    try:
        os.mkdir(raw_data_path + "_tg")
    except:
        pass

    processed_data_path = raw_data_path + "_tg"

    for i in tqdm(range(data_num)):
        md_path = raw_data_path + '/{}.pkl'.format(i)
        with open(md_path, 'rb+') as f:
            md_data = pickle.load(f)
        init_data = MDdata_to_initdata(md_data)

    # dump init_data in processed_data_path
        with open(processed_data_path + '/{}.pkl'.format(i), 'wb+') as f:
            pickle.dump(init_data, f)
