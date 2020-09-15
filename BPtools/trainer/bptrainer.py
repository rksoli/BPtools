import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import time

from BPtools.core.bpmodule import BPModule


class BPTrainer:
    def __init__(self, *args, **kwargs):
        self.model = None
        self.criterion = None
        self.epochs = None
        self.losses: Dict = {}

    @staticmethod
    def elapsed_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        elapsed_milisecs = int((elapsed_time - elapsed_mins * 60 - elapsed_secs) * 1000)
        return elapsed_mins, elapsed_secs, elapsed_milisecs

    def fit(
            self,
            model: BPModule,
            train_dataloader: Optional[DataLoader] = None,
            validation_dataloader: Optional[DataLoader] = None
    ):
        # do the training
        model.trainer = self
        self.model = model
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
