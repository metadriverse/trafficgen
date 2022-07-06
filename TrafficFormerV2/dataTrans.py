import math
import os
import pickle
import sys

import numpy as np
import tensorflow as tf
from tqdm import trange

roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/objects_of_interest':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/future/id':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/state':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/valid':
        tf.io.FixedLenFeature([80, 16], tf.int64, default_value=None),
    'traffic_light_state/future/x':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/y':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/future/z':
        tf.io.FixedLenFeature([80, 16], tf.float32, default_value=None),
    'traffic_light_state/current/id':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/id':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}
scenario_features = {
    'scenario/id':
        tf.io.FixedLenFeature([1], tf.string, default_value=None)
}
features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)
features_description.update(scenario_features)


MAX_ROAD_LEN = 251  # the longest road contains up to 1933 points!
MAX_ROAD_NUM = 660  # the maximum road num is 651
MAX_NBRS_NUM = 127
CHANNELS = 13
PAST_LEN = 11
FUTURE_LEN = 80
TIMEFRAME = 91


def yaw_to_y(angle):
    angle = trans_angle(angle)
    angle_to_y = angle - np.pi / 2
    angle_to_y = -1 * angle_to_y

    return angle_to_y


def yaw_to_theta(angles, theta):
    theta = trans_angle(theta)
    tranformed = []
    for angle in angles:
        angle -= theta
        angle = trans_angle(angle)
        if angle >= np.pi:
            angle -= 2 * np.pi
        tranformed.append(angle)
    return tranformed

    # transform angles into 0~2pi


def trans_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle >= 2 * np.pi:
        angle -= 2 * np.pi
    return angle


def transform_coord(coords, angle):
    x = coords[:, 0]
    y = coords[:, 1]
    x_transform = math.cos(angle) * x - math.sin(angle) * y
    y_transform = math.cos(angle) * y + math.sin(angle) * x
    output_coords = np.stack((x_transform, y_transform), axis=-1)
    if coords.shape[1] == 3:
        output_coords = np.concatenate((output_coords, coords[:, 2:]), axis=-1)
    return output_coords


# def divide_lanes(rid, xyz, valid, rtype):
#     feat_list = []
#     id_set = np.unique(rid)
#     for _id in id_set:
#         if _id == -1:
#             continue
#         arg = np.where(rid == _id)[0]
#
#         if len(arg) <= 1:
#             one_point = (xyz[arg[0], :], xyz[arg[0], :], rtype[arg[0], 0], _id, valid[arg[0], 0])
#             feat = np.hstack(one_point).astype(np.float32).reshape(1, -1)
#             feat = np.pad(feat, [(0, MAX_ROAD_LEN - 2), (0, 0)], mode='constant')  # shape: MAX_ROAD_LEN, c
#             feat_list.append(np.expand_dims(feat, axis=0))
#             continue
#
#         for k in range(len(arg) // (MAX_ROAD_LEN + 1) + 1):
#             L, R = k * MAX_ROAD_LEN, min((k + 1) * MAX_ROAD_LEN, len(arg))
#             selected_xyz = xyz[arg[L:R], :]
#             selected_valid = valid[arg[L:R], :]
#             selected_type = rtype[arg[L:R], :]
#             # selected_dir = rdir[arg[L:R], :] # TODO: current have no idea how to deal with direction
#             '''
#             channel:
#             0-5: x y z (in vector mode)
#             6 vector type
#             7: vector id
#             8: vector validity
#             '''
#             vector_xy = np.concatenate((selected_xyz[:-1, :], selected_xyz[1:, :]), -1)
#             vector_type = selected_type[1:, ...]
#             vector_id = np.full((len(vector_type), 1), _id)
#             vector_valid = np.expand_dims(selected_valid[1:, 0] * selected_valid[:-1, 0], 1)
#             feat = np.concatenate([vector_xy, vector_type, vector_id, vector_valid], axis=-1).astype(np.float32)
#             feat[feat[:, -1] == 0, :] = 0
#             feat = np.pad(feat, [(0, MAX_ROAD_LEN - R + L), (0, 0)], mode='constant')  # shape: MAX_ROAD_LEN, c
#             feat_list.append(np.expand_dims(feat, axis=0))
#
#     lane_feat = np.concatenate(feat_list, axis=0)
#     valid_len = lane_feat.shape[0]
#     lane_feat = np.pad(lane_feat, [(0, MAX_ROAD_NUM - valid_len), (0, 0), (0, 0)],
#                        mode='constant')  # shape: MAX_ROAD_NUM, MAX_ROAD_LEN, c
#     return lane_feat, valid_len
if __name__=="__main__":
    # rootpath='/mnt/lustre/share/zhangqihang/WOD/'
    rootpath = sys.argv[1]
    dirname = 'parsed'
    g = os.walk(os.path.join(rootpath, 'tf_example'))
    for path, dir_list, file_list in g:

        print(path)

        tail = path[-2::-1].find('/')
        cnt = 0
        new_dir = os.path.join(rootpath, dirname, path[-tail - 1:])
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        l = len(file_list)
        for i in trange(l):
            file_name = file_list[i]
            if 'tfrecord' not in file_name:
                continue

            file_path = os.path.join(path, file_name)
            dataset = tf.data.TFRecordDataset(file_path, compression_type='')
            for j, data in enumerate(dataset.as_numpy_iterator()):
                parsed = tf.io.parse_single_example(data, features_description)
                d = dict()
                res = dict()

                for k, v in parsed.items():
                    v = v.numpy()
                    d[k] = v
                # -----------------------------rotation start------------------------------
                p_c_f = np.zeros([MAX_NBRS_NUM + 1, TIMEFRAME, CHANNELS], dtype=np.float32)


                def my_concat(l):
                    return np.squeeze(np.concatenate(list(np.expand_dims(item, axis=-1) for item in l), axis=-1))


                p_c_f[:, :10] = my_concat((d['state/past/x'], d['state/past/y'], d['state/past/z'],
                                           d['state/past/velocity_x'], d['state/past/velocity_y'],
                                           d['state/past/bbox_yaw'], d['state/past/length'],
                                           d['state/past/width'], d['state/past/height'],
                                           d['state/type'].repeat(10).reshape(128, 10),
                                           d['state/past/timestamp_micros'] / 1e6,
                                           d['state/past/valid'], d['state/id'].repeat(10).reshape(128, 10)))

                p_c_f[:, 10] = my_concat((d['state/current/x'], d['state/current/y'], d['state/current/z'],
                                          d['state/current/velocity_x'], d['state/current/velocity_y'],
                                          d['state/current/bbox_yaw'], d['state/current/length'],
                                          d['state/current/width'], d['state/current/height'],
                                          d['state/type'].reshape(128, 1),
                                          d['state/current/timestamp_micros'] / 1e6,
                                          d['state/current/valid'], d['state/id'].reshape(128, 1)))

                p_c_f[:, 11:] = my_concat((d['state/future/x'], d['state/future/y'], d['state/future/z'],
                                           d['state/future/velocity_x'], d['state/future/velocity_y'],
                                           d['state/future/bbox_yaw'], d['state/future/length'],
                                           d['state/future/width'], d['state/future/height'],
                                           d['state/type'].repeat(80).reshape(128, 80),
                                           d['state/future/timestamp_micros'] / 1e6,
                                           d['state/future/valid'], d['state/id'].repeat(80).reshape(128, 80)))

                p_c_f = p_c_f[d['state/id'] != -1]
                valid_obj_num = p_c_f.shape[0]
                p_c_f = np.pad(p_c_f, ((0, MAX_NBRS_NUM + 1 - valid_obj_num), (0, 0), (0, 0)), 'constant')

                for i, v in enumerate(d['state/is_sdc']):
                    if v == 1:
                        sdc_index = i
                        break

                center = p_c_f[sdc_index, 10, :2].copy()
                ego_yaw = p_c_f[sdc_index, 10, 5].copy()
                theta = yaw_to_y(ego_yaw).astype(np.float32)

                p_c_f = p_c_f.reshape(-1, CHANNELS)
                p_c_f[:, 0] = p_c_f[:, 0] - center[0]
                p_c_f[:, 1] = p_c_f[:, 1] - center[1]
                p_c_f[:, :2] = transform_coord(p_c_f[:, :2], theta)
                p_c_f[:, 3:5] = transform_coord(p_c_f[:, 3:5], theta)
                p_c_f[:, 5] = yaw_to_theta(p_c_f[:, 5], ego_yaw)
                p_c_f[p_c_f[:, -2] == 0, :] = 0
                p_c_f = p_c_f.reshape(MAX_NBRS_NUM + 1, TIMEFRAME, CHANNELS)

                res['ego_p_c_f'] = p_c_f[sdc_index, :, :]
                res['nbrs_p_c_f'] = np.delete(p_c_f, sdc_index, 0)
                res['tracks_to_train'] = np.all(res['nbrs_p_c_f'][:, :11, -2], axis=-1).astype(np.int)

                # --------------------processing lanes---------------------------------------------------#
                rid = d['roadgraph_samples/id']
                xyz = d['roadgraph_samples/xyz']
                rdir = d['roadgraph_samples/dir']
                rtype = d['roadgraph_samples/type']
                valid = d['roadgraph_samples/valid']

                xyz[:, 0] = xyz[:, 0] - center[0]
                xyz[:, 1] = xyz[:, 1] - center[1]
                xyz[:, :2] = transform_coord(xyz[:, :2], theta)
                rdir[:, :2] = transform_coord(rdir[:, :2], theta)
                res['lane'] = np.concatenate([rid, xyz, rdir, rtype, valid], -1).astype(np.float32)

                # -----------processing traffic lights---------------------------#
                traf_p_c_f = np.zeros((TIMEFRAME, 16, 6), dtype=np.float32)

                traf_p_c_f[:10, :, 0] = d['traffic_light_state/past/id']
                traf_p_c_f[:10, :, 1] = d['traffic_light_state/past/x']
                traf_p_c_f[:10, :, 2] = d['traffic_light_state/past/y']
                traf_p_c_f[:10, :, 3] = d['traffic_light_state/past/z']
                traf_p_c_f[:10, :, 4] = d['traffic_light_state/past/state']
                traf_p_c_f[:10, :, 5] = d['traffic_light_state/past/valid']

                traf_p_c_f[10, :, 0] = d['traffic_light_state/current/id']
                traf_p_c_f[10, :, 1] = d['traffic_light_state/current/x']
                traf_p_c_f[10, :, 2] = d['traffic_light_state/current/y']
                traf_p_c_f[10, :, 3] = d['traffic_light_state/current/z']
                traf_p_c_f[10, :, 4] = d['traffic_light_state/current/state']
                traf_p_c_f[10, :, 5] = d['traffic_light_state/current/valid']

                traf_p_c_f[11:, :, 0] = d['traffic_light_state/future/id']
                traf_p_c_f[11:, :, 1] = d['traffic_light_state/future/x']
                traf_p_c_f[11:, :, 2] = d['traffic_light_state/future/y']
                traf_p_c_f[11:, :, 3] = d['traffic_light_state/future/z']
                traf_p_c_f[11:, :, 4] = d['traffic_light_state/future/state']
                traf_p_c_f[11:, :, 5] = d['traffic_light_state/future/valid']

                traf_p_c_f[:, :, 1] = traf_p_c_f[:, :, 1] - center[0]
                traf_p_c_f[:, :, 2] = traf_p_c_f[:, :, 2] - center[1]
                traf_p_c_f = traf_p_c_f.reshape(-1, 6)
                traf_p_c_f[:, 1:3] = transform_coord(traf_p_c_f[:, 1:3], theta)
                traf_p_c_f[traf_p_c_f[:, -1] == 0, :] = -1
                traf_p_c_f = traf_p_c_f.reshape(TIMEFRAME, 16, 6)

                res['traf_p_c_f'] = traf_p_c_f

                # -----------properties------------------#
                lis = [sdc_index] + list(range(sdc_index)) + list(range(sdc_index + 1, 128))
                res['tracks_to_predict'] = d['state/tracks_to_predict'][lis]
                res['objects_of_interest'] = d['state/objects_of_interest'][lis]
                res['theta'] = theta
                res['id'] = d['scenario/id']
                res['center'] = center

                # dump data into pkl
                p = os.path.join(rootpath, dirname, path[-tail - 1:], f'{cnt}.pkl')

                with open(p, 'wb') as f:
                    pickle.dump(res, f)
                cnt += 1
