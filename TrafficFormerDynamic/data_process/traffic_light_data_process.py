import numpy as np


class WaymoTrafficLight:
    TOTAL_TYPE = 3

    Unknown = 0,
    Arrow_Stop = 1
    Arrow_Caution = 2
    Arrow_Go = 3
    Stop = 4
    Caution = 5
    Go = 6
    Flashing_Stop = 7
    Flashing_Caution = 8

    ONE_HOT_DIM = 2

    @staticmethod
    def is_stop_type(type):
        return True if int(type) in [WaymoTrafficLight.Arrow_Stop,
                                     WaymoTrafficLight.Stop,
                                     WaymoTrafficLight.Flashing_Stop] else False

    @staticmethod
    def is_go_type(type):
        return True if int(type) in [WaymoTrafficLight.Go,
                                     WaymoTrafficLight.Arrow_Go,
                                     ] else False

    @staticmethod
    def is_caution(type):
        return True if int(type) in [WaymoTrafficLight.Caution,
                                     WaymoTrafficLight.Arrow_Caution,
                                     WaymoTrafficLight.Flashing_Caution] else False
    @staticmethod
    def is_unknown(type):
        return True if type==0. or type == 0 else False

    @staticmethod
    def one_hot(type):
        if WaymoTrafficLight.is_caution(type):
            return np.array([0, 0])
        elif WaymoTrafficLight.is_go_type(type):
            return np.array([0, 1])
        elif WaymoTrafficLight.is_stop_type(type):
            return np.array([1, 0])
        elif WaymoTrafficLight.is_unknown(type):
            return np.array([1, 1])
        else:
            raise ValueError("Unknown traffic light type: {}".format(type))