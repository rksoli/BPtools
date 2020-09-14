# import torch
import torch.nn as nn
import torch.optim as optim
# import numpy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


class BPModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    def forward(self, *args):
        return super().forward(*args)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self
                             ) -> Optional[Union[optim.Optimizer, Sequence[optim.Optimizer], Dict, Sequence[Dict],
                                                 Tuple[List, List]]]:
        raise NotImplementedError

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
    ) -> None:
        optimizer.step()

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: optim.Optimizer, optimizer_idx: int):
        optimizer.zero_grad()

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

        self.train()

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


