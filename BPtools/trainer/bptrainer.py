import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import time

from BPtools.core.bpmodule import BPModule
from BPtools.trainer.connetcors.model_connector import ModelConnector


class BPTrainer:
    def __init__(self, *args, **kwargs):
        self.model: BPModule = None  # kwargs["model"] if "model" in kwargs else args[0]
        self.criterion = None

        # CONNECTORS
        self.model_connector = ModelConnector(self)

        # self.optim_configuration = None  # nem tudom jó ötlet-e
        self.epochs: int = kwargs["epochs"] if "epochs" in kwargs else None
        self.losses: Dict = {"train": [], "valid": []}
        self.dataloaders: Dict = {"train": [], "valid": [], "test": []}

    @staticmethod
    def elapsed_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        elapsed_milisecs = int((elapsed_time - elapsed_mins * 60 - elapsed_secs) * 1000)
        return elapsed_mins, elapsed_secs, elapsed_milisecs

    def print(self, epoch, elapsed_time):
        print('epoch: ', epoch, 'time: ', elapsed_time[0], 'mins', elapsed_time[1], 'secs', elapsed_time[2],
              'mili secs')
        print('train loss: ', self.losses["train"][-1])
        print('valid loss: ', self.losses["valid"][-1])

    def load_data(self):
        pass

    def fit(
            self,
            model: BPModule,
            train_dataloader: Optional[DataLoader] = None,
            validation_dataloader: Optional[DataLoader] = None
    ):
        # do the training
        self.model_connector.connect(model)
        self.model = model.to("cuda")
        self.criterion = model.criterion
        optim_configuration = model.configure_optimizers()

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.training_step(optim_configuration)
            self.model.validation_step()
            end_time = time.time()
            epoch_time = self.elapsed_time(start_time, end_time)
            # TODO: save model params
            # TODO: epoch print

        self.model.test_step()


