import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from tensorboardX import SummaryWriter

import time

from BPtools.core.bpmodule import BPModule
from BPtools.core.bpdatamodule import BPDataModule
from BPtools.trainer.connetcors.model_connector import ModelConnector
from BPtools.trainer.connetcors.data_connector import DataConnector


class BPTrainer:
    def __init__(self, *args, **kwargs):
        self.model: BPModule = None  # kwargs["model"] if "model" in kwargs else args[0]
        self.datamodule: BPDataModule = None

        # pointer to callable loss nn.Module
        self.criterion = kwargs["criterion"] if "criterion" in kwargs else None

        # CONNECTORS
        self.model_connector = ModelConnector(self)
        self.data_conncector = DataConnector(self)

        # tensor board writer
        self.writer = SummaryWriter(logdir='log/losses')

        # self.optim_configuration = None  # nem tudom jó ötlet-e
        self.epochs: int = kwargs["epochs"] if "epochs" in kwargs else None
        self.losses: Dict = {"train": [], "valid": []}
        self.dataloaders: Dict = {"train": None, "valid": None, "test": None}

        # bool
        self.is_data_loaded = False
        self.is_data_prepared = False


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
        if not self.is_data_loaded:
            self.model.setup()
            self.is_data_loaded = True

    def logger(self, step):
        self.writer.add_scalar('train loss', self.losses["train"][-1], step)
        self.writer.add_scalar('valid loss', self.losses["valid"][-1], step)
        self.writer.add_scalars('train and valid losses', {'train': self.losses["train"][-1],
                                                           'valid': self.losses["valid"][-1]}, step)

    def setup(self, train_dataloader, validation_dataloader, datamodule):
        # TODO: setup függvény a fit() beállításához
        self.data_conncector.attach_data(train_dataloader, validation_dataloader, datamodule)
        # self.data_conncector.prepare_data(model=self.model)
        has_setup = isinstance(self.datamodule, BPDataModule) and self.datamodule.has_setup_fit
        if not has_setup and self.datamodule is not None:
            self.datamodule.setup()

    def fit(
            self,
            model: BPModule,
            train_dataloader: Optional[DataLoader] = None,
            validation_dataloader: Optional[DataLoader] = None,
            datamodule: Optional[BPDataModule] = None
    ):
        # do the training
        self.model_connector.connect(model)
        self.setup(train_dataloader, validation_dataloader, datamodule)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        # self.model.setup()
        #######
        # DATA CONNECTOR HASZNÁLATA
        #######
        # self.data_conncector.attach_data(train_dataloader, validation_dataloader, datamodule)
        optim_configuration = model.configure_optimizers()

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.training_step(optim_configuration, epoch)
            self.model.validation_step(epoch)
            end_time = time.time()
            epoch_time = self.elapsed_time(start_time, end_time)
            # TODO: save model params
            # TODO: epoch print
            self.print(epoch, epoch_time)
            self.logger(epoch)

        self.model.test_step()
        self.writer.close()


