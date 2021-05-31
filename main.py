from BPtools.trainer.bptrainer import BPTrainer
# from BPtools.utils.models import VariationalAutoEncoder, AdvAE, VarDecoderConv1d_3, VarEncoderConv1d, VAEDataModul, Discriminator
from BPtools.utils.models import *
from BPtools.core.bpmodule import BPModule
from BPtools.metrics.criterions import KLD_MSE_loss_variational_autoencoder
import inspect
import torch

# encoder = VarEncoderConv1d(2, 60, 10)
encoder = EncoderBN(2, 60, 10)
decoder = VarDecoderConv1d_3(2, 60, 10)
disc = Discriminator(10, 2)
print(sum(p.numel() for p in disc.parameters() if p.requires_grad))
print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))


my_model_1 = VariationalAutoEncoder(encoder, decoder)
# my_model_1.load_state_dict(torch.load(
#     'model_parameters.pt'))
#
# print(my_model_1.state_dict())
#
# encoder.load_state_dict(my_model_1.encoder.state_dict())
# decoder.load_state_dict(my_model_1.decoder.state_dict())
my_model = AdvAE(encoder, decoder, disc)
print(sum(p.numel() for p in my_model.parameters() if p.requires_grad))

my_dm = VAEDataModul(path="c:/repos/full_data/X_Yfull_dataset.npy", split_ratio=0.1)
# print(inspect.getsource(CustomLossVAE(1,1).forward))
Trainer = BPTrainer(epochs=200000, criterion=KLD_MSE_loss_variational_autoencoder(0.1, 0), name="WITH3GAUSS2advvae_n_disc_lay2")
print(isinstance(my_model, BPModule))
Trainer.fit(model=my_model, datamodule=my_dm)
# print(my_model.parameters())
# print(my_model.state_dict())
