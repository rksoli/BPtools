
class ModelConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def connect(self, model):
        model.trainer = self.trainer

    def get_model(self):
        return self.trainer.model
