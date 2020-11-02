from BPtools.utils.vehicle import *
from typing import Any, List, Optional, Tuple, Union
import pandas as pd


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
        self.data = data
        self.csv_file_name = csv_file_name
        # grid related properties
        self.length = kwargs["length"] if "length" in kwargs else None
        self.width = kwargs["width"] if "width" in kwargs else None
        self.deltaX = kwargs["deltaX"] if "deltaX" in kwargs else None
        self.deltaY = kwargs["deltaY"] if "deltaY" in kwargs else None
        self.number_of_lanes = kwargs["number_of_lanes"] if "number_of_lanes" in kwargs else None

        if self.vehicle_objects is None:
            if self.
    def load_csv_file(self):
        self.vehicle_dataset = VehicleDataset(self.csv_file_name)

    def create_vehicle_objects(self):
        self.vehicle_objects = self.vehicle_dataset.create_vehicle_objects()

