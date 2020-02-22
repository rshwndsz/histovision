
class BaseTester(object):
    def __init__(self):
        pass

    def load(self):
        """Load model weights"""
        pass

    def forward(self, images):
        """Forward pass"""
        raise NotImplementedError

    def start(self):
        """Testing loop"""
        raise NotImplementedError
