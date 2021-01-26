from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import *
from BPtools.core.bpmodule import BPModule
from BPtools.metrics.criterions import KLD_BCE_loss_2Dvae

encoder = Encoder2D2()
t = torch.rand((15,1,16,128))
dr = nn.Dropout2d()
z=encoder(dr(t))
print(z[0].shape)
print(sum(p.numel() for p in encoder.parameters() if p.requires_grad))
decoder = Decoder2D2(155)
print(sum(p.numel() for p in decoder.parameters() if p.requires_grad))
print(decoder(z[0]).shape)
vae = VAE2D(encoder=encoder, decoder=decoder)

dm = GridVAE_DataModul(path='D:/dataset/grids/31349.0_11.npy', split_ratio=0.2)

trainer = BPTrainer(epochs=20000, criterion=KLD_BCE_loss_2Dvae(0.1))
trainer.fit(model=vae, datamodule=dm)
