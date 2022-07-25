import sys

from tqdm import tqdm

try:
    import tensorflow as tf
except ImportError:
    pass
from drivingforce.TrafficTranformer.utils import scenario_pb2
import os
import pickle
import numpy as np

SAMPLE_NUM = 10
LANE_DIM = 4
TIME_SAMPLE = 3  # sample 64 time step, 64*3 = 192
BATCH_SIZE = 91  # 64*3 < 192, ok


def yaw_to_y(angles):
    ret = []
    for angle in angles:
        angle = trans_angle(angle)
        angle_to_y = angle - np.pi / 2
        angle_to_y = -1 * angle_to_y
        ret.append(angle_to_y)
    return np.array(ret)


def yaw_to_theta(angles, thetas):
    """
    In time horizon
    """
    ret = []
    for theta, angle in zip(thetas, angles):
        theta = trans_angle(theta)
        angle -= theta
        angle = trans_angle(angle)
        if angle >= np.pi:
            angle -= 2 * np.pi
        ret.append(angle)
    return np.array(ret)

    # transform angles into 0~2pi


def trans_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def transform_coord(coords, angle):
    x = coords[..., 0]
    y = coords[..., 1]
    x_transform = np.cos(angle) * x - np.sin(angle) * y
    y_transform = np.cos(angle) * y + np.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)

    if coords.shape[1] == 3:
        output_coords = np.concatenate((output_coords, coords[:, 2:]), axis=-1)
    return output_coords


def extract_tracks(f, sdc_index,pred_list):
    #agents = np.zeros([len(f), BATCH_SIZE, 9])
    agents = np.zeros([len(f), BATCH_SIZE, 9])
    pred_mask = np.zeros(len(f))
    pred_mask[pred_list]=1
    sdc = f[sdc_index]
    sdc_x = np.array([state.center_x for state in sdc.states])
    sdc_y = np.array([state.center_y for state in sdc.states])
    sdc_yaw = np.array([state.heading for state in sdc.states])
    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
    for i in range(len(f)):
        x = np.array([state.center_x for state in f[i].states]) - sdc_x
        y = np.array([state.center_y for state in f[i].states]) - sdc_y
        pos = transform_coord(np.concatenate([np.expand_dims(x, -1), np.expand_dims(y, -1)], axis=-1), sdc_theta)
        # z = [state.center_z for state in f[i].states]
        l = np.array([[state.length] for state in f[i].states])
        w = np.array([[state.width] for state in f[i].states])
        # h = [state.height for state in f[i].states]
        head = np.expand_dims(yaw_to_theta(np.array([state.heading for state in f[i].states]), sdc_yaw), axis=-1)

        vx = np.array([state.velocity_x for state in f[i].states])
        vy = np.array([state.velocity_y for state in f[i].states])
        velocity = transform_coord(np.concatenate([np.expand_dims(vx, -1), np.expand_dims(vy, -1)], axis=-1), sdc_theta)

        valid = np.array([[state.valid] for state in f[i].states])
        t = np.expand_dims(np.repeat(f[i].object_type, len(valid)), axis=-1)

        #time_sample = int(len(pos)/BATCH_SIZE)  # some case may do not have 20s
        #time_sample = min(time_sample, TIME_SAMPLE)
        #agents[i] = np.concatenate((pos, velocity, head, l, w, t, valid), axis=-1)[::time_sample, ...][:BATCH_SIZE]

        agents[i] = np.concatenate((pos, velocity, head, l, w, t, valid), axis=-1)[:BATCH_SIZE]
    ego = agents[sdc_index]
    others = np.delete(agents, sdc_index, axis=0)
    pred_mask = np.delete(pred_mask,sdc_index,axis=0)
    #others = np.transpose(others,[1,0,2])
    # others[abs(others[..., 0] > 80), -1] = 0
    # others[abs(others[..., 1] > 80), -1] = 0

    return ego, others,pred_mask


def extract_dynamic(f, sdc):
    #dynamics = np.zeros([BATCH_SIZE, 32, 6])
    dynamics = []
    #time_sample = min(int(len(sdc.states)/BATCH_SIZE), TIME_SAMPLE)
    # sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    # sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_x = np.array([state.center_x for state in sdc.states])[:BATCH_SIZE]
    sdc_y = np.array([state.center_y for state in sdc.states])[:BATCH_SIZE]
    sdc_yaw = np.array([state.heading for state in sdc.states])[:BATCH_SIZE]
    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)

    for i in range(BATCH_SIZE):
        #states = f[i * time_sample].lane_states
        states = f[i].lane_states
        traf_list = []
        for j in range(len(states)):
            traf = np.zeros(6)
            traf[0] = states[j].lane
            traf[1:4] = transform_coord(
                np.array([[states[j].stop_point.x - sdc_x[i], states[j].stop_point.y - sdc_y[i], 0]]),
                np.array([sdc_theta[i]]))
            if states[j].state in [1,4,7]:
                state_ = 1 # stop
            elif states[j].state in [2,5,8]:
                state_ = 2 # caution
            elif states[j].state in [3,6]:
                state_ = 3 # go
            else:
                state_ = 0  # unknown
            traf[4] = state_
            traf[5] = 1 if states[j].state else 0
            traf_list.append(traf)
        dynamics.append(traf_list)
    return dynamics


def extract_poly(message):
    x = [i.x for i in message]
    y = [i.y for i in message]
    z = [i.z for i in message]
    coord = np.stack((x, y, z), axis=1)

    return coord


def down_sampling(line, type=0):
    # if is center lane
    point_num = len(line)

    ret = []

    if point_num < SAMPLE_NUM or type == 1:
        for i in range(0, point_num):
            ret.append(line[i])
    else:
        for i in range(0,point_num,SAMPLE_NUM):
            ret.append(line[i])

    return ret


def extract_boundaries(fb):
    b = []
    for k in range(len(fb)):
        b.append(fb[k].boundary_feature_id)
    return b


def extract_neighbors(fb):
    nbs = []
    for k in range(len(fb)):
        nbs.append(fb[k].feature_id)
    return nbs


def extract_center(f):

    f = f.lane

    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    poly = [np.insert(x,2,f.type) for x in poly]

    return poly


def extract_line(f):

    f = f.road_line
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = f.type + 5
    poly = [np.insert(x,2,type) for x in poly]
    return poly


def extract_edge(f):

    f = f.road_edge
    poly = down_sampling(extract_poly(f.polyline)[:, :2])
    type = 15 if f.type == 1 else 16
    poly = [np.insert(x,2,type) for x in poly]

    return poly


def extract_stop(f):

    f = f.stop_sign
    ret = np.array([f.position.x, f.position.y,17])

    return [ret]


def extract_crosswalk(f):

    f = f.crosswalk
    poly = down_sampling(extract_poly(f.polygon)[:, :2],1)
    poly = [np.insert(x,2,18) for x in poly]
    return poly


def extract_bump(f):

    f = f.speed_bump
    poly = down_sampling(extract_poly(f.polygon)[:, :2], 1)
    poly = [np.insert(x, 2, 19) for x in poly]
    return poly


def extract_map(f):
    maps = []
    #nearbys = dict()
    for i in range(len(f)):
        id = f[i].id

        if f[i].HasField('lane'):
            line = extract_center(f[i])

        elif f[i].HasField('road_line'):
            line = extract_line(f[i])

        elif f[i].HasField('road_edge'):
            line = extract_edge(f[i])

        elif f[i].HasField('stop_sign'):
            line = extract_stop(f[i])

        elif f[i].HasField('crosswalk'):
            line = extract_crosswalk(f[i])

        elif f[i].HasField('speed_bump'):
            line = extract_bump(f[i])
        else:
            continue

        line = [np.insert(x, 3, id) for x in line]
        maps = maps + line

    return np.array(maps)


def transform_coordinate_map(map, sdc):
    """
    Every frame is different
    """
    time_sample = min(int(len(sdc.states) / BATCH_SIZE), TIME_SAMPLE)
    sdc_x = np.array([state.center_x for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_y = np.array([state.center_y for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_yaw = np.array([state.heading for state in sdc.states])[::time_sample, ...][:BATCH_SIZE]
    sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
    pos = np.stack([sdc_x, sdc_y], axis=-1)
    ret = np.zeros(shape=(BATCH_SIZE, *map.shape))
    for i in range(BATCH_SIZE):
        ret[i] = map
        ret[i][..., :2] = transform_coord(ret[i][..., :2] - pos[i],
                                           np.expand_dims(sdc_theta[i], -1))


    # ret[abs(ret[:, :, :, 1]) > 80,-1] =0
    # ret[abs(ret[:, :, :, 2]) > 80, -1] = 0
    valid_ret = np.sum(ret[...,-1],-1)
    lane_mask = valid_ret.astype(bool)
    ret[ret[...,-1]==0,:] = 0

    return ret,lane_mask

def add_traff_to_lane(scene):
    traf = scene['traf_p_c_f']
    lane = scene['lane']
    traf_buff = np.zeros([*lane.shape[:2]])
    for i in range(BATCH_SIZE):
        lane_i_id = lane[i,:,0,-1]
        for a_traf in traf[i]:
            lane_id = a_traf[0]
            state = a_traf[-2]
            lane_idx = np.where(lane_i_id==lane_id)
            traf_buff[i,lane_idx] = state
    return traf_buff





def parse_data(inut_path, output_path, pre_fix=None):
    MAX=100000
    cnt = 0
    scenario = scenario_pb2.Scenario()
    file_list = os.listdir(inut_path)
    for file in tqdm(file_list):
        file_path = os.path.join(inut_path, file)
        if not 'tfrecord' in file_path:
            continue
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for j, data in enumerate(dataset.as_numpy_iterator()):
            try:
                if pre_fix=='None':
                    p = os.path.join(output_path, '{}.pkl'.format(cnt))
                else:
                    p = os.path.join(output_path, '{}_{}.pkl'.format(pre_fix, cnt))

                scenario.ParseFromString(data)
                scene = dict()
                scene['id'] = scenario.scenario_id
                sdc_index = scenario.sdc_track_index

                pred_list = [i.track_index for i in scenario.tracks_to_predict]
                scene['ego_p_c_f'], scene['nbrs_p_c_f'],scene['pred_list'] = extract_tracks(scenario.tracks, sdc_index,pred_list)

                ego = scenario.tracks[sdc_index]
                scene['traf_p_c_f'] = extract_dynamic(scenario.dynamic_map_states, ego)


                scene['lane'] = extract_map(scenario.map_features)

                time_sample = min(int(len(ego.states) / BATCH_SIZE), TIME_SAMPLE)
                sdc_x = np.array([state.center_x for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_y = np.array([state.center_y for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_yaw = np.array([state.heading for state in ego.states])[::time_sample, ...][:BATCH_SIZE]
                sdc_theta = yaw_to_y(sdc_yaw).astype(np.float32)
                pos = np.stack([sdc_x, sdc_y], axis=-1)
                scene['sdc_theta'] = sdc_theta
                scene['sdc_pos'] = pos

            except:
                print(f'fail to parse {cnt},continue')
                continue

            with open(p, 'wb') as f:
                pickle.dump(scene, f)

            cnt += 1
            if cnt > MAX:
                break
        if cnt > MAX:
            break
    return


if __name__ == '__main__':
    """
    Usage: devide the source data to several pieces, like 10 dirs each containing 100 rf record data, out_put to on dir
    for x in 10
        mkdir raw_x
        move 0-100 raw_x
    then you put data to 10 dir
    for x in 10
        nohup python trans20 ./raw_x ./scenario x > x.log 2>&1 &
    NOTE: 3 *x* to change when manually repeat !!!!!
    ```tail -n 100 -f x.log```  to vis progress of process x
    
    After almost 30 minutes, all cases are stored in dir scenario
    Then run ```nohup python unify.py scenario > unify.log 2>&1 &``` to unify the name      
    
    ls -l scenario | wc -l 
    output: 70542   
    
    Some data may be broken
    """
    raw_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]
    pre_fix = sys.argv[3]
    # raw_data_path = ".\\data"
    # processed_data_path = ".\\debug_data"
    # pre_fix = str(uuid.uuid4())
    #  parse raw data from input path to output path,
    #  there is 1000 raw data in google cloud, each of them produce about 500 pkl file
    parse_data(raw_data_path, processed_data_path, pre_fix)
