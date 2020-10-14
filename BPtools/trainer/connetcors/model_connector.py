
class ModelConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def connect(self, model):
        model.trainer = self.trainer
        self.trainer.model = model

    def get_model(self):
        return self.trainer.model
