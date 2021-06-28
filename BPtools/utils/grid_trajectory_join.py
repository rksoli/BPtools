from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
# import modin.pandas as pd
# from sodapy import Socrata
from PIL import Image
from matplotlib import cm


class DataProcess:
    def __init__(self,
                 path,
                 delta_T1=6.0,
                 delta_T2=None,
                 break_threshold=0.8,
                 overtaking=False,
                 braking=False
                 ):
        self.delta_T1 = delta_T1
        self.delta_T2 = delta_T2 if delta_T2 is not None else delta_T1
        self.break_threshold = break_threshold
        self.overtaking = overtaking
        self.braking = braking
        self.path = path
        self.__mode = "ob" if overtaking and braking else "o" if overtaking and not braking else "b" if not (
                    overtaking or not braking) else None
        self.time_reg = ["31349.0", "67669.0", "75649.0"]
        self.maxnum_reg = [21, 19, 10]

    @property
    def path_grids(self):
        return self.path + '/grids'

    @property
    def path_trajs(self):
        return self.path + '/trajs'

    def build_dataset(self):
        for T, N in zip(self.time_reg, self.maxnum_reg):
            for n in range(N):
                grid_Tn = np.load(self.path_grids +"/"+T+"_"+str(n)+".npy", allow_pickle=True)
                trajs_Tn = np.load(self.path_trajs +"/traj"+T+"_"+str(n)+".npy", allow_pickle=True)



if __name__ == "__main__":
    data = DataProcess(path='D:/dataset')
    print(data.path_grids, data.path_trajs)

