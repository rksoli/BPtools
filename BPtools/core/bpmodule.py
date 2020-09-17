import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import BPtools.trainer.bptrainer as bpt


class BPModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BPModule, self).__init__()

        # pointer to the trainer
        self.trainer: bpt.BPTrainer = None

        # pointer to the logger
        self.logger = None

        # pointer to callable loss nn.Module
        self.criterion = None

    def print(self, *args, **kwargs) -> None:
        print(*args, **kwargs)

    def forward(self, *args):
        return super().forward(*args)

    def load_data(self, path):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """

        # Multiple optimizers (e.g.: GANs)
            def training_step(self, batch, batch_idx, optimizer_idx):
                if optimizer_idx == 0:
                    # do training_step with encoder
                if optimizer_idx == 1:
                    # do training_step with decoder

        ---
        def training_step(self, batch, batch_idx):
                x, y, z = batch
                # implement your own
                out = self(x)
                loss = self.loss(out, x)
                # TrainResult auto-detaches the loss after the optimization steps are complete
                result = pl.TrainResult(minimize=loss)
        """
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(
            self
    ) -> Optional[Union[optim.Optimizer, Sequence[optim.Optimizer], Dict, Sequence[Dict], Tuple[List, List]]]:
        """
        # most cases
                def configure_optimizers(self):
                    opt = Adam(self.parameters(), lr=1e-3)
                    return opt
                # multiple optimizer case (e.g.: GAN)
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    return generator_opt, disriminator_opt
                # example with learning rate schedulers
                def configure_optimizers(self):
                    generator_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    disriminator_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    discriminator_sched = CosineAnnealing(discriminator_opt, T_max=10)
                    return [generator_opt, disriminator_opt], [discriminator_sched]
                # example with step-based learning rate schedulers
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    gen_sched = {'scheduler': ExponentialLR(gen_opt, 0.99),
                                 'interval': 'step'}  # called after each training step
                    dis_sched = CosineAnnealing(discriminator_opt, T_max=10) # called every epoch
                    return [gen_opt, dis_opt], [gen_sched, dis_sched]
                # example with optimizer frequencies
                # see training procedure in `Improved Training of Wasserstein GANs`, Algorithm 1
                # https://arxiv.org/abs/1704.00028
                def configure_optimizers(self):
                    gen_opt = Adam(self.model_gen.parameters(), lr=0.01)
                    dis_opt = Adam(self.model_disc.parameters(), lr=0.02)
                    n_critic = 5
                    return (
                        {'optimizer': dis_opt, 'frequency': n_critic},
                        {'optimizer': gen_opt, 'frequency': 1}
                    )
        """
        raise NotImplementedError

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: optim.Optimizer,
        optimizer_idx: int,
    ) -> None:
        optimizer.step()

        """
         Alternating schedule for optimizer steps (i.e.: GANs)
                def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                                   second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    # update generator opt every 2 steps
                    if optimizer_idx == 0:
                        if batch_idx % 2 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()
                    # update discriminator opt every 4 steps
                    if optimizer_idx == 1:
                        if batch_idx % 4 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()
        """

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


