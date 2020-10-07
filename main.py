from BPtools.trainer.bptrainer import BPTrainer
from BPtools.utils.models import VariationalAutoEncoder, VarDecoderConv1d_3, VarEncoderConv1d
from BPtools.core.bpmodule import BPModule
from BPtools.metrics. criterions import CustomLossVAE

encoder = VarEncoderConv1d(2, 60, 10)
decoder = VarDecoderConv1d_3(2, 60, 10)
my_model = VariationalAutoEncoder(encoder, decoder)
data_loader = None  # TODO: dataloader kimásolása a training base-ből a dataloading.py-ba
Trainer = BPTrainer(epochs=80000, criterion=CustomLossVAE(0.1, 1.0))
print(isinstance(my_model, BPModule))
Trainer.fit(model=my_model)
# print(my_model.parameters())
# print(my_model.state_dict())
