from BPtools import *

my_model = BPModule()
data_loader = None  # TODO: dataloader kimásolása a training base-ből a dataloading.py-ba
Trainer = BPTrainer(model=my_model)
Trainer.fit(model=my_model)
print(my_model.parameters())
print(Trainer.model.parameters())
