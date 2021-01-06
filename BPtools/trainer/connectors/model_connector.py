
class ModelConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def connect(self, model):
        model.trainer = self.trainer
        self.trainer.model = model

        if model.losses_keys is not None:
            for keys in model.losses_keys:
                self.trainer.losses[keys] = []

    def get_model(self):
        return self.trainer.model
