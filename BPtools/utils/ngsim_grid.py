from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
# import modin.pandas as pd
# from sodapy import Socrata
from PIL import Image
from matplotlib import cm


class OccupancyGrid:
    def __init__(self,
                 vehicle_objects: List[VehicleData] = None,
                 vehicle_dataset: VehicleDataset = None,
                 data: pd.DataFrame = None,
                 csv_file_name: str = None,
                 **kwargs):

        # data related properties
        self.vehicle_objects: List[VehicleData] = vehicle_objects
        self.vehicle_dataset: VehicleDataset = vehicle_dataset
        self._data = None
        self.data = data
        self.csv_file_name = csv_file_name
        # grid related properties
        self.length = kwargs["length"] if "length" in kwargs else None
        self.width = kwargs["width"] if "width" in kwargs else None
        self.deltaX = kwargs["deltaX"] if "deltaX" in kwargs else None
        self.deltaY = kwargs["deltaY"] if "deltaY" in kwargs else None
        self.number_of_lanes = kwargs["number_of_lanes"] if "number_of_lanes" in kwargs else None
        self.y_min_max = []
        self.x_min_max = []

        if csv_file_name is not None:
            self.load_vd()
            # self.create_vehicle_objects()
        else:
            if self.vehicle_dataset is not None:
                self.create_vehicle_objects()
            else:
                # TODO 4: load pd.Dataframe from csv
                # TODO 4: create v.o. from df
                pass

    @property
    def data(self):
        if self._data is not None:
            return self._data
        elif self.vehicle_dataset is not None:
            return self.vehicle_dataset.all_data
        else:
            return

    @data.setter
    def data(self, data):
        self._data = data

    def load_vd(self):
        self.vehicle_dataset = VehicleDataset(self.csv_file_name)

    def create_vehicle_objects(self):
        self.vehicle_objects = self.vehicle_dataset.create_vehicle_objects()

    def grid_prepare(self):
        self.data["Time_Start"] = self.data["Global_Time"] / 100 - self.data["Frame_ID"] - 1.11343e+10
        # print(self.data["Time_Start"].value_counts())
        self.y_min_max = [self.data["Local_Y"].min(), self.data["Local_Y"].max() + 10.0]
        self.x_min_max = [self.data["Local_X"].min() - 2.0, self.data["Local_X"].max() + 2.0]
        # print(self.y_min_max, self.x_min_max)
        for T, time_start in self.data.groupby("Time_Start"):
            for f_id, frame in time_start.groupby("Frame_ID"):
                x_num = int((self.x_min_max[1] - self.x_min_max[0]) // self.deltaX + 1)
                y_num = int((self.y_min_max[1] - self.y_min_max[0]) // self.deltaY + 1)
                # Todo(fontos): property x_num, y_num
                pic_array = np.zeros((x_num, y_num))
                for _, vehicle in frame.groupby("Vehicle_ID"):
                    X = (vehicle["Local_X"] - self.x_min_max[0]) // self.deltaX
                    Y = (vehicle["Local_Y"] - self.y_min_max[0]) // self.deltaY
                    x_1 = int(X - 0.5 * vehicle["v_Width"] / self.deltaX)
                    y_1 = int(Y)
                    x_2 = int(X + 0.5 * vehicle["v_Width"] / self.deltaX)
                    y_2 = int(Y + vehicle["v_length"] // self.deltaY)
                    pic_array[x_1:x_2, y_1:y_2] = 1.0
                if f_id > 100:
                    im = Image.fromarray(np.uint8(cm.gist_earth(pic_array) * 255))
                    im.save(str(f_id) + ".png")

    def grid_data_for_ae(self, size=(32, 256)):
        self.data["Time_Start"] = self.data["Global_Time"] / 100 - self.data["Frame_ID"] - 1.11343e+10
        # Konvenció: az origó a közepétől eggyel jobbra és feljebb index
        origo = (size[0] // 2 + 1, size[1] // 2 + 1)
        # idősor csoportosítás kezdő időpont szerint.
        for T, group_time_start in self.data.groupby("Time_Start"):
            print("T: ", T)
            # tj=0
            # if T == 75649:
            #     pass
            # else:
            #     continue

            data_T = []
            i = 1

            # adott idősor csoportosítás: ego járművek
            for ego_v_ID, group_ego_vehicles in group_time_start.groupby("Vehicle_ID"):
                data_ego = []
                ego = VehicleData(np.array(group_ego_vehicles))
                # Todo(Data): itt az ego-ból kellene kinyerni a trajektóriát
                print("\tEgo ID: ", ego_v_ID)
                print("\tframes: ", ego.size)

                # adott idősor csoportosítás: frame ID szerint
                for f_id, group_frame in group_time_start.groupby("Frame_ID"):
                    neighbourhood = np.zeros(size, dtype=np.uint8)
                    # EGO
                    # Ego pozíciója f_id-ben
                    x_ego_f_id = ego[("Local_X", f_id)]
                    y_ego_f_id = ego[("Local_Y", f_id)]
                    if not x_ego_f_id:
                        # ha nincs benne az ego ebben az f_id-ben - azaz f_id nem része az egonak, akkor
                        # False a visszatérési érték a __getitem__ függvénynek
                        # ugrás a következő f_id-re
                        continue

                    # Ego  (L,W)
                    ego_dims = ego[("v_dims", f_id)]

                    # Ego berakása
                    x_1 = origo[0] - int(0.5 * ego_dims[1] / self.deltaX)
                    x_2 = origo[0] + int(0.5 * ego_dims[1] / self.deltaX)
                    y_1 = origo[1]
                    y_2 = origo[1] + int(ego_dims[0] / self.deltaY)
                    neighbourhood[x_1:x_2, y_1:y_2] = 1

                    # a frameben található járművek csoportosítása (szomszédság)
                    for v_ID, group_vehicles in group_frame.groupby("Vehicle_ID"):
                        if v_ID == ego_v_ID:
                            # nem kell még egyszer berakni az egot
                            # Bár azt is meg lehet csinálni hogy itt teszem bele általánosan.
                            continue

                        v = VehicleData(np.array(group_vehicles))
                        x_v_f_id = v[("Local_X", f_id)]
                        y_v_f_id = v[("Local_Y", f_id)]
                        v_dims = ego[("v_dims", f_id)]
                        # Ego koordinátarendszerben
                        x_v_f_id = x_v_f_id - x_ego_f_id
                        y_v_f_id = y_v_f_id - y_ego_f_id
                        x_1 = int(x_v_f_id // self.deltaX) + origo[0] - int(0.5 * v_dims[1] / self.deltaX)
                        x_2 = int(x_v_f_id // self.deltaX) + origo[0] + int(0.5 * v_dims[1] / self.deltaX)
                        y_1 = int(y_v_f_id // self.deltaX) + origo[1]
                        y_2 = int(y_v_f_id // self.deltaX) + origo[1] + int(v_dims[0] / self.deltaY)
                        if x_1 < 0 or x_2 >= size[0] or y_1 < 0 or y_2 >= size[1]:
                            continue
                        neighbourhood[x_1:x_2, y_1:y_2] = 1

                    # Egy adott ego_v_ID-hez és f_id-hez tartozó grid elkészült
                    # Show

                    data_ego.append(neighbourhood)
                    # im = Image.fromarray(np.uint8(cm.gist_earth(neighbourhood) * 255))
                    # im.save("ego_" + str(ego_v_ID) + "_f_" + str(f_id) + ".png")
                # Todo(Data): itt kellene a data_ego mellé appendelni az ego teljes trajektóriáját is. Lehet hogy sok
                #  lesz, ezért egy teljesen különálló függvény gyárthatná le csak a trajektóriákat ugyan abban a
                #  sorrendben és struktúrában
                data_T.append(data_ego)
                print("\tVehicles: ", len(data_T))

                if len(data_T) == 100:
                    np.save(str(T) + '_' + str(i), np.array(data_T))
                    data_T = []
                    i = i + 1
                # ha nem érte el a 100-at, ki kéne még menteni
            # maradék data_T kimentése, ha van
            if len(data_T) > 0:
                np.save(str(T) + '_' + str(i), np.array(data_T))

    def trajectory_for_grid(self):
        self.data["Time_Start"] = self.data["Global_Time"] / 100 - self.data["Frame_ID"] - 1.11343e+10
        # Konvenció: az origó a közepétől eggyel jobbra és feljebb index
        # delete
        size = []
        # origo = (size[0] // 2 + 1, size[1] // 2 + 1)
        # idősor csoportosítás kezdő időpont szerint.
        for T, group_time_start in self.data.groupby("Time_Start"):
            print("T: ", T)

            data_T = []
            i = 1

            # adott idősor csoportosítás: ego járművek
            for ego_v_ID, group_ego_vehicles in group_time_start.groupby("Vehicle_ID"):
                data_ego = []
                ego = VehicleData(np.array(group_ego_vehicles))
                ego.lane_changing()
                # Todo(Data): itt az ego-ból kellene kinyerni a trajektóriát
                print("\tEgo ID: ", ego_v_ID)
                print("\tframes: ", ego.size)
                ego_traj = np.concatenate((ego.x.reshape(-1,1), ego.y.reshape(-1,1), ego.v.reshape(-1,1),
                                           ego.lane_id.reshape(-1,1), ego.change_lane[:,0].reshape(-1,1)), axis=1)
                # for ego_lane_ID, group_ego_v_lane in group_ego_vehicles.groupby("Lane_ID"):
                #     print("\t\tlane: ", ego_lane_ID)
                # print(ego_traj.shape)
                # print(ego_traj[0:10])
                # print(ego_traj[60:70])


                # Todo(Data): itt kellene a data_ego mellé appendelni az ego teljes trajektóriáját is. Lehet hogy sok
                #  lesz, ezért egy teljesen különálló függvény gyárthatná le csak a trajektóriákat ugyan abban a
                #  sorrendben és struktúrában
                data_T.append(ego_traj)
                print("\tVehicles: ", len(data_T))

                if len(data_T) == 100:
                    np.save("traj" + str(T) + '_' + str(i), np.array(data_T))
                    data_T = []
                    i = i + 1
                # ha nem érte el a 100-at, ki kéne még menteni
            # maradék data_T kimentése, ha van
            if len(data_T) > 0:
                np.save("traj" + str(T) + '_' + str(i), np.array(data_T))


if __name__ == "__main__":
    '''
    # ,Vehicle_ID,Frame_ID,Total_Frames,Global_Time,
    # Local_X,Local_Y,Global_X,Global_Y,v_length,v_Width,v_Class,v_Vel,v_Acc,Lane_ID,
    # O_Zone,D_Zone,Int_ID,Section_ID,Direction,Movement,Preceding,Following,Space_Headway,Time_Headway,Location
    '''

    # og = OccupancyGrid(csv_file_name='../../../full_data/i-80.csv')
    # print(og.data)
    # client = Socrata("data.transportation.gov", None)
    # results = client.get("8ect-6jqj")#, limit=2000)
    # results_df = pd.DataFrame.from_records(results)
    # print(results_df.head())
    grid = OccupancyGrid(csv_file_name='../../../full_data/us-101.csv', deltaX=0.5, deltaY=0.5)
    grid.grid_data_for_ae()
    grid.trajectory_for_grid()
