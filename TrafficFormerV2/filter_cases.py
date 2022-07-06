from drivingforce.TrafficTranformer.data_process.map_data_process import WaymoMap
import os
import pickle

if __name__ == "__main__":
    import sys
    # data_path = "../debug"
    data_path = sys.argv[1]
    for index in range(70000):
        file_path = os.path.join(data_path, f'{index}.pkl')
        with open(file_path, 'rb+') as f:
            data = pickle.load(f)
        map = WaymoMap(data, data['traf_p_c_f'])
        try:
            map.maps
        except:
            os.remove(file_path)