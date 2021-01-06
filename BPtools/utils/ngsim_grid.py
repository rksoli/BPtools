from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from sodapy import Socrata
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
                    y_2 = int(Y + vehicle["v_length"]//self.deltaY)
                    pic_array[x_1:x_2, y_1:y_2] = 1.0
                if f_id > 100:
                    im = Image.fromarray(np.uint8(cm.gist_earth(pic_array) * 255))
                    im.save(str(f_id) + ".png")


        print(5)




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
    grid = OccupancyGrid(csv_file_name='../../../full_data/i-80.csv', deltaX=0.5, deltaY=0.5)
    grid.grid_prepare()
