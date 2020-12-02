from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd
from sodapy import Socrata


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

        if csv_file_name is not None:
            self.load_vd()
            self.create_vehicle_objects()
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


if __name__ == "__main__":
    # og = OccupancyGrid(csv_file_name='../../../full_data/i-80.csv')
    # print(og.data)
    client = Socrata("data.transportation.gov", None)
    results = client.get("8ect-6jqj")#, limit=2000)
    results_df = pd.DataFrame.from_records(results)
    print(results_df.head())
