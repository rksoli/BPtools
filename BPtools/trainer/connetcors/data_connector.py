from typing import Any, List, Optional, Tuple, Union
from BPtools.core.bpdatamodule import BPDataModule


class DataConnector:
    # TODO: megnézni, hogyan lesz ez működőképes egy DataModul osztállyal
    def __init__(self, trainer):
        self.trainer = trainer

    def connect(self, **kwargs):
        pass

    def prepare_data(self, model=None):
        if self.trainer.datamodule is not None:
            self.trainer.datamodule.prepare_data()
        elif model is not None:
            model.prepare_data()
            self.trainer.is_data_loaded = True
        else:
            Exception('You did not pass a model or a datamodule therefore data preparation is failed')
        self.trainer.is_data_prepared = True

    def attach_datamodule(self, datamodule: Optional[BPDataModule]) -> None:
        if datamodule is not None:
            # connect trainer to datamodule
            self.trainer.datamodule = datamodule
            datamodule.trainer = self.trainer

            datamodule.prepare_data()
            datamodule.setup()
            # TODO: check if pointer is passed. If not, then NoneType is passed to the trainer and must be fixed
            self.trainer.dataloaders = {"train": datamodule.train_dataloader(),
                                        "valid": datamodule.val_dataloader(),
                                        "test": datamodule.test_dataloader()}
            self.trainer.is_data_loaded = True

    def attach_data(self, train_dataloader, val_dataloader, datamodule):
        if isinstance(train_dataloader, BPDataModule):
            datamodule = train_dataloader
            train_dataloader = None

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if (train_dataloader is not None or val_dataloader is not None) and datamodule is not None:
            raise Exception(
                'You cannot pass train_dataloader or val_dataloaders to trainer.fit if you supply a datamodule'
            )
        self.attach_dataloaders(train_dataloader, val_dataloader)
        self.attach_datamodule(datamodule)

    def attach_dataloaders(self, train_dataloader=None, val_dataloader=None, test_dataloader=None):
        if train_dataloader is not None:
            self.trainer.dataloaders["train"] = train_dataloader
        if val_dataloader is not None:
            self.trainer.dataloaders["valid"] = train_dataloader
        if test_dataloader is not None:
            self.trainer.dataloaders["test"] = train_dataloader
        if None not in (train_dataloader, val_dataloader):
            self.trainer.is_data_loaded = True
