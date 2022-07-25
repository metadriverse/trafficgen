import numpy as np
import os
import pickle


class WaymoAgentInfo:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    # vehicle, pedestrian, cyclist, others and unset
    AGENT_TYPE_ONE_HOT = np.eye(3)
    ONE_HOT_DIM = 2

    def __init__(self, feature):
        # index of xy,v,lw,yaw,type,valid
        self.position = feature[:2]
        self.velocity = feature[2:4]
        self.length_width = feature[4:6]
        self.heading = np.arctan2(feature[6],feature[7])
        self.end_point = feature[8:10]
        # self.type = 'none'
        # if feature[-3] == 1:
        #     self.type = "car"
        # elif feature[-2] == 1:
        #     self.type = "ped"
        # elif feature[-1] == 1:
        #     self.type == "cyc"

        # self.type = self.get_type_from_one_hot(feature[8:])

    @staticmethod
    def heading_direction(heading):
        return np.sin(heading), np.cos(heading)

    @staticmethod
    def one_hot_index(type):
        if int(type) == WaymoAgentInfo.UNSET or int(type) == WaymoAgentInfo.OTHER:
            index = 3
        else:
            index = int(type) - 1
        return index

    @staticmethod
    def one_hot(type):
        return WaymoAgentInfo.AGENT_TYPE_ONE_HOT[WaymoAgentInfo.one_hot_index(type)]

    # @staticmethod
    # def get_type_from_one_hot(type):
    #     type[0] = int(type[0])
    #     type[1] = int(type[1])
    #     if type == [0, 0]:
    #         return WaymoAgentInfo.VEHICLE
    #     elif index == [0, 1]:
    #         return WaymoAgentInfo.PEDESTRIAN
    #     elif index == [1, 0]:
    #         return WaymoAgentInfo.CYCLIST
    #     else:
    #         raise ValueError("No such agent type: {}".format(type))


class WaymoScene:
    """
    Comparing to WaymoMap which records static information, this class records dynamic information
    """
    ONE_CASE_SCENE_NUM = 192
    MAX_LENGTH = 50.
    END_MAX = 110.
    MAX_VELOCITY = 30 # m/s

    def __init__(self, case_data, batch_size=16, max_agent_num=32):
        """
        The sample frequency must be same as the traffic light sample frequency
        :sample_frequency: usually batch size
        """
        self.max_agent_num = max_agent_num
        self.batch_size = batch_size
        self.agent_data, self.agent_mask= self.process_data(case_data)


    def process_data(self, case_data):
        """
        Index of agent in case data xy,vxvy,yaw,lw,type,valid, after processing: xy,v_norm,lw, yaw,type
        """
        agent_data = []
        agent_masks = []


        # all_feat includes ego_car and nbrs car
        all_feat = np.concatenate([np.expand_dims(case_data['ego_p_c_f'], 0), case_data['nbrs_p_c_f']], 0)
        # for every timestamp
        ind_ = list(range(0,190,10))
        for i in ind_:
            a_scene = all_feat[:, i]

            # filter invalid agent and delete valid dim
            valid_mask = a_scene[:, -1] == 1.
            range_mask = (abs(a_scene[:,0])<self.MAX_LENGTH)*(abs(a_scene[:,1])<self.MAX_LENGTH)

            type_mask = a_scene[:, -2] == 1.

            mask = valid_mask*range_mask*type_mask

            a_scene = a_scene[mask][..., :-1]
            valid_num = a_scene.shape[0]
            # use cos,sin to represent heading

            headings = -a_scene[:, 4][..., np.newaxis]
            sin_h = np.sin(headings)
            cos_h = np.cos(headings)
            a_scene = np.concatenate([a_scene, sin_h,cos_h], -1)
            a_scene = np.delete(a_scene, [4, 7], axis=-1)

            # sort agent in top-down. left-right manner
            ego = a_scene[0][np.newaxis, ...]
            a_scene = a_scene[1:]
            raster = np.floor(a_scene[:, :2])
            raster[:, 1] = raster[:, 1] // 10
            raster = np.concatenate([raster, a_scene], -1)
            y_index = np.argsort(-raster[:, 1])
            raster = raster[y_index]
            y_set = np.unique(raster[:, 1])[::-1]
            for y in y_set:
                ind = np.argwhere(raster[:, 1] == y)[:, 0]
                ys = raster[ind]
                x_index = np.argsort(ys[:, 0])
                raster[ind] = ys[x_index]
            scene = np.delete(raster, [0, 1], axis=-1)
            cur_scene = scene

            cur_scene = np.concatenate([ego, cur_scene], 0)

            cur_scene = cur_scene[:self.max_agent_num]
            cur_scene = np.pad(cur_scene, ([0, self.max_agent_num - cur_scene.shape[0]], [0, 0]))
            agent_data.append(cur_scene)

            valid_num = min(valid_num, self.max_agent_num)
            agent_mask = np.zeros(self.max_agent_num)
            agent_mask[:valid_num]=1
            agent_masks.append(agent_mask)

        agent_data = np.array(agent_data)
        agent_masks = np.array(agent_masks)


        agent_nums = np.sum(agent_masks,axis=-1)
        for i in range(agent_data.shape[0]):
            agent_num = int(agent_nums[i])
            permut_idx = np.random.permutation(np.arange(1,agent_num))-1

            agent_data[i,1:agent_num] = agent_data[i,1:agent_num][permut_idx]
            agent_masks[i, 1:agent_num] = agent_masks[i, 1:agent_num][permut_idx]


        return agent_data,agent_masks

    def get_agent_information(self, time_step, agent_id):
        """
        This function retrieve agent information from the scene
        Note: time_step is chosen from post-processing data, which contains self.selected_time_step_index steps
        """
        assert 0 <= time_step < self.batch_size, "time step out of scope"
        assert 0 <= agent_id < self.valid_agent_num[time_step], "agent id out of scope"
        feature = self.agent_data[time_step][agent_id]
        return WaymoAgentInfo(feature)


if __name__ == "__main__":
    data_path = "../debug_data"
    maps = []
    for index in range(78):
        file_path = os.path.join(data_path, f'{index}.pkl')
        with open(file_path, 'rb+') as f:
            data = pickle.load(f)
        scene = WaymoScene(data, batch_size=32)
        info = scene.get_agent_information(0, 0)
        maps.append(scene)
    # print(max([map.line_num() for map in maps]))
