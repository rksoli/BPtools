import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import time
from pandas import DataFrame

from BPtools.core.bpmodule import BPModule
from BPtools.core.bpdatamodule import BPDataModule
from BPtools.trainer.connectors.model_connector import ModelConnector
from BPtools.trainer.connectors.data_connector import DataConnector


class BPTrainer:
    def __init__(self, *args, **kwargs):
        self.model: BPModule = None
        self.datamodule: BPDataModule = None

        # pointer to callable loss nn.Module
        self.criterion = kwargs["criterion"] if "criterion" in kwargs else None

        # CONNECTORS
        self.model_connector = ModelConnector(self)
        self.data_conncector = DataConnector(self)

        # tensor board writer
        name = kwargs["name"] if "name" in kwargs else ''
        self.name = name
        self.writer = SummaryWriter('log_' + name + '/losses')

        # self.optim_configuration = None  # nem tudom jó ötlet-e
        self.epochs: int = kwargs["epochs"] if "epochs" in kwargs else 100
        self.losses: Dict = {"train": [], "valid": []}
        self.min_loss = float("inf")
        # TODO 3: Check if dictionay is the best way
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
        for key in self.losses.keys():
            print(key + ' :', round(self.losses[key][-1], 4))

    def load_data(self):
        if not self.is_data_loaded:
            self.model.setup()
            self.is_data_loaded = True

    def logger(self, step):
        # Todo: ha megy rendesen, akkor végig iterálni a losses dictionaryn és kész
        self.writer.add_scalar('train loss', self.losses["train"][-1], step)
        self.writer.add_scalar('valid loss', self.losses["valid"][-1], step)
        self.writer.add_scalars('train and valid losses', {'train': self.losses["train"][-1],
                                                           'valid': self.losses["valid"][-1]}, step)
        for key in self.losses.keys():
            if key not in ["train", "valid"]:
                self.writer.add_scalar(key, self.losses[key][-1], step)

        if self.losses["valid"][-1] < self.min_loss:
            self.min_loss = self.losses["valid"][-1]
            torch.save(self.model.state_dict(), 'log_' + self.name + "/model_state_dict_" + self.name)

    def setup(self, train_dataloader, validation_dataloader, datamodule):
        # TODO 3: a model "információk" lekérése
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
        print('device: ', device)
        self.model = model.to(device)
        optim_configuration = model.configure_optimizers()

        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.training_step(optim_configuration, epoch)
            self.model.validation_step(epoch)
            end_time = time.time()
            epoch_time = self.elapsed_time(start_time, end_time)
            # TODO 2: save model params
            self.print(epoch, epoch_time)
            self.logger(epoch)

        self.writer.close()
        self.save_losses()

    def save_losses(self):
        dict = {'epoch': list(range(1, self.epochs + 1))}
        dict.update(self.losses)
        losses = DataFrame(dict)
        losses.to_csv('losses.csv', index=False)
        losses.to_excel('losses.xlsx', index=False)


