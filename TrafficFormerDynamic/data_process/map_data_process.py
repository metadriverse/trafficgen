import numpy as np
from drivingforce.TrafficTranformer.data_process.traffic_light_data_process import WaymoTrafficLight
import pickle
import os


class WaymoLineType:
    TOTAL_LINE_TYPE = 5
    LANE_CENTER_FREEWAY = 1
    LANE_CENTER_SURFACE = 2
    LANE_CENTER_BIKE = 3
    BROKEN_SINGLE_WHITE = 6
    SOLID_SINGLE_WHITE = 7
    SOLID_DOUBLE_WHITE = 8
    BROKEN_SINGLE_YELLOW = 9
    BROKEN_DOUBLE_YELLOW = 10
    SOLID_SINGLE_YELLOW = 11
    SOLID_DOUBLE_YELLOW = 12
    PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    StopSign = 17
    Crosswalk = 18
    SpeedBump = 19
    UNKNOWN = 0

    ONE_HOT_DIM = 2

    @staticmethod
    def is_lane(type):
        return True if int(type) in [WaymoLineType.LANE_CENTER_SURFACE, WaymoLineType.LANE_CENTER_FREEWAY, WaymoLineType.LANE_CENTER_BIKE] else False

    @staticmethod
    def is_unknown(type):
        return True if int(type) in [WaymoLineType.UNKNOWN] else False

    @staticmethod
    def is_road_edge(type):
        return True if int(type) in [WaymoLineType.ROAD_EDGE_BOUNDARY, WaymoLineType.ROAD_EDGE_MEDIAN] else False

    @staticmethod
    def is_road_line(type):
        return True if int(type) in [WaymoLineType.BROKEN_SINGLE_WHITE,
                                     WaymoLineType.SOLID_SINGLE_WHITE,
                                     WaymoLineType.SOLID_DOUBLE_WHITE,
                                     WaymoLineType.BROKEN_SINGLE_YELLOW,
                                     WaymoLineType.BROKEN_DOUBLE_YELLOW,
                                     WaymoLineType.SOLID_SINGLE_YELLOW,
                                     WaymoLineType.SOLID_DOUBLE_YELLOW,
                                     WaymoLineType.PASSING_DOUBLE_YELLOW] else False

    @staticmethod
    def is_broken_yellow(type):
        return True if int(type) in [
            WaymoLineType.BROKEN_SINGLE_YELLOW,
            WaymoLineType.BROKEN_DOUBLE_YELLOW] else False

    @staticmethod
    def is_solid_yellow(type):
        return True if int(type) in [
            WaymoLineType.SOLID_SINGLE_YELLOW,
            WaymoLineType.SOLID_DOUBLE_YELLOW,
            WaymoLineType.PASSING_DOUBLE_YELLOW] else False

    @staticmethod
    def is_broken_white(type):
        return True if int(type) in [WaymoLineType.BROKEN_SINGLE_WHITE] else False

    @staticmethod
    def is_solid_white(type):
        return True if int(type) in [
            WaymoLineType.SOLID_SINGLE_WHITE,
            WaymoLineType.SOLID_DOUBLE_WHITE] else False

    @staticmethod
    def one_hot(type):
        if WaymoLineType.is_lane(type):
            return np.array([0, 0])
        elif WaymoLineType.is_solid_white(type):
            return np.array([0, 1])
        elif WaymoLineType.is_broken_white(type):
            return np.array([1, 0])
        elif WaymoLineType.is_solid_yellow(type) or WaymoLineType.is_broken_yellow(type):
            return np.array([1, 1])
        # else:
        #     raise ValueError("Unknown type: {}".format(type))

class WaymoLine:
    MAX_LENGTH = 80.

    def __init__(self, line_feature):
        """
        Line feature: [rid, x, y, type, valid]
        Points vector [sample point num * [this_point_x, this_point_y]]
        Type is one of the WaymoLineType
        """
        self.id = np.max(line_feature[:,-1])
        self.type = np.max(line_feature[:,-2])
        self.points = line_feature[..., :2]
        #self.line_is_valid = np.sum(line_feature[..., 4])>0
        self.line_is_valid = True
        self.valid = line_feature[:,-2].astype(bool)
        #
        # if np.argwhere(self.points == np.array([0.0, 0.0])).any():
        #     self.points = np.zeros_like(self.points)

    @staticmethod
    def get_line_points(feature, sample_point_num):
        """
        Get line points and do filtering
        """
        points_vector = np.zeros([sample_point_num, 2], dtype=np.float32)
        lis = [1, 2]
        valid = feature[:, 8]
        feature = feature[:, lis]
        feature = feature[valid == 1]
        point_num = len(feature)
        if point_num < sample_point_num:
            points_vector[:point_num] = feature[:, :2]
            # repeat last point
            points_vector[point_num:] = np.tile(feature[-1, :2], (sample_point_num - point_num, 1))
        else:
            interval = 1.0 * (point_num - 1) / sample_point_num
            selected_point_index = [int(np.round(i * interval)) for i in range(1, sample_point_num - 1)]
            selected_point_index = [0] + selected_point_index + [point_num - 1]
            selected_point = feature[selected_point_index, :]
            points_vector[:, :2] = selected_point[:, :]
        return points_vector

    def line_type(self):
        return self.line_type()

    def get_line_vector(self, need_type=False):
        """
        This will return a training used line vector
        NOTE: the length is sample_point_num-1
        """
        # points_vector = np.zeros([len(self.points) - 1, 4 + WaymoLineType.ONE_HOT_DIM if need_type else 4],
        #                          dtype=np.float32)
        points_vector = np.zeros([len(self.points) - 1,5])
        points_vector[:, 0:2] = self.points[:-1]
        points_vector[:, 2:4] = self.points[1:]
        points_vector[:, 4] = self.type

        validity = self.valid[:-1]*self.valid[1:]
        points_vector[validity==0,:] = self.MAX_LENGTH # a unifying place out of region
        #points_vector/=self.MAX_LENGTH
        # if need_type:
        #     points_vector[:, 4:] = WaymoLineType.one_hot(self.type)
        return points_vector


class WaymoMap:
    ONE_CASE_MAP_NUM = 192

    def __init__(self, case_data,
                 traffic_light_info,
                 batch_size=64,
                 need_line_type=True,
                 need_traffic_light=False,
                 need_edge=False):
        """
        Only contain center line and road line, not edge and bike lane, etc.
        """
        filter = [i for i in range(0, self.ONE_CASE_MAP_NUM, int(self.ONE_CASE_MAP_NUM / batch_size))]
        self.traffic_light_info = traffic_light_info[filter]
        self.traffic_light_info[...,1:3] /= WaymoLine.MAX_LENGTH
        self.need_traffic_light = need_traffic_light
        self.need_line_type = need_line_type
        self.batch_size = batch_size
        map_data,_= self.process_case(case_data)
        self.maps, self.lane_masks = self.get_map_vectors(map_data)
        #self.edges, _ = self.get_map_vectors(edge_data)

    def process_case(self, case_data):
        # np.concatenate([rid, xyz, rdir, rtype, valid], -1).astype(np.float32)
        # self.lane_mask = case_data['lane_mask']
        maps = []
        edges = []
        lanes = case_data["lane"]
        step_size = int(self.ONE_CASE_MAP_NUM/self.batch_size)
        for map_i,lane in enumerate(lanes):
            if map_i % step_size != 0:
                continue
            # lane is actually map, this is an name error
            current_map = []
            #current_edge = []
            for one_line in lane:
                line = WaymoLine(one_line)
                # if not line.line_is_valid:
                #     continue
                # if not WaymoLineType.is_lane(line.type) and not WaymoLineType.is_road_line(
                #         line.type):
                # if WaymoLineType.is_road_edge(line.type):
                #     current_edge.append(line)
                # else:
                current_map.append(line)
            maps.append(current_map)
            #edges.append(current_edge)
        return maps, edges

    def line_num(self):
        return np.array([len(self.maps[0]) for _ in range(self.batch_size)])

    def get_map_vectors(self, maps):
        rett = []
        max_lane_num = 0
        for time_step in range(self.batch_size):
            ret = []
            traffic_light_id = self.traffic_light_info[time_step][:, 0]
            max_lane_num = max(len(maps[time_step]), max_lane_num)
            for line in maps[time_step]:
                line_info = line.get_line_vector(self.need_line_type)
                # if self.need_traffic_light:
                #     ind = np.argwhere(traffic_light_id == line.id)
                #     if ind.shape[0] == 0:
                #         type = np.zeros([len(line_info), WaymoTrafficLight.ONE_HOT_DIM])
                #     else:
                #         type = np.tile(WaymoTrafficLight.one_hot(self.traffic_light_info[time_step][ind[0][0], -2]),
                #                        (len(line_info), 1))
                #     line_info = np.concatenate([line_info, type], axis=-1)
                ret.append(np.expand_dims(line_info, axis=0))
            rett.append(np.concatenate(ret, axis=0))
        rettt = []
        for map in rett:
            after_pad = np.zeros([max_lane_num, map.shape[1], map.shape[2]], dtype=map.dtype)
            after_pad[:len(map), ...] = map
            rettt.append(np.expand_dims(after_pad, axis=0))
        res = np.concatenate(rettt, axis=0)
        valid = np.sum(res[...,0],axis=-1)
        lane_mask = np.ones([*res.shape[:2]])
        lane_mask[valid==0]=0
        return np.concatenate(rettt, axis=0), lane_mask


if __name__ == "__main__":
    import sys
    import time
    data_path = "../debug_data"
    # data_path = sys.argv[1]
    start = time.time()
    for index in range(70):
        file_path = os.path.join(data_path, f'{index}.pkl')
        with open(file_path, 'rb+') as f:
            data = pickle.load(f)
        map = WaymoMap(data, data['traf_p_c_f'], batch_size=32)
        try:
            m = map.maps
            msk = map.lane_masks
        except:
            print("Can not parse: {}".format(index))
    print(time.time()-start)
