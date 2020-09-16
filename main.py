from BPtools import *

my_model = BPModule()
Trainer = BPTrainer(model=my_model)
print(my_model.parameters())
print(Trainer.model.parameters())
