import torch


class eval_mode(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.prev = self.model.training
        self.model.train(False)

    def __exit__(self, *args):
        self.model.train(self.prev)
        return False
