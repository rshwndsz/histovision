# Python STL
import logging
# PyTorch
import torch
from torch.backends import cudnn
# Advanced config
import hydra
# Local
import histovision.testers

# Get root logger
logger = logging.getLogger('root')


@hydra.main("config/config.yaml")
def test(cfg):
    # Set constants
    # device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        cfg.device = "cuda:0"  # <<< Note: Single-GPU
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # Faster convolutions at the expense of memory
    cudnn.benchmark = cfg.training.cudnn_benchmark

    # Get tester
    tester = eval(cfg.tester)(cfg)
    # Start testing loop
    tester.start()


if __name__ == "__main__":
    test()
