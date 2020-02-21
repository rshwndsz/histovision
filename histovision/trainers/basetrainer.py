
class BaseTrainer(object):
    def __init__(self):
        pass

    def forward(self, images, targets):
        raise NotImplementedError

    def iterate(self, epoch, phase):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError
