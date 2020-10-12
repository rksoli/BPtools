from typing import Any, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch import Tensor, FloatTensor


class BPDataModule:
    def __init__(self):
        self.trainer = None

        # Private attrs to keep track of whether or not data hooks have been called yet
        self._has_prepared_data = False
        self._has_setup_fit = False
        self._has_setup_test = False

    @property
    def has_prepared_data(self):
        """Return bool letting you know if datamodule.prepare_data() has been called or not.
        Returns:
            bool: True if datamodule.prepare_data() has been called. False by default.
        """
        return self._has_prepared_data

    @property
    def has_setup_fit(self):
        """Return bool letting you know if datamodule.setup('fit') has been called or not.
        Returns:
            bool: True if datamodule.setup('fit') has been called. False by default.
        """
        return self._has_setup_fit

    @property
    def has_setup_test(self):
        """Return bool letting you know if datamodule.setup('test') has been called or not.
        Returns:
            bool: True if datamodule.setup('test') has been called. False by default.
        """
        return self._has_setup_test

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, Tensor, FloatTensor]:
        raise NotImplementedError

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader], Tensor, FloatTensor]:
        raise NotImplementedError

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader], Tensor, FloatTensor]:
        raise NotImplementedError
