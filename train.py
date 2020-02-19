# Imports
# Python STL
import os
from pathlib import Path
import sys
# Data Science
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.backends.cudnn as cudnn
# Hydra
import hydra
# Local
from histovision.shared import log
from histovision.models.unet import model
from histovision.trainers.BinaryTrainer import BinaryTrainer as Trainer

# Constants
# Path to current directory `pwd`
_HERE = os.path.dirname(__file__)


@hydra.main(config_path="config.yaml")
def train(cfg):
    # Create root logger
    logger = log.setup_logger('root')
    # Faster convolutions at the expense of memory
    cudnn.benchmark = cfg.cudnn_benchmark
    # Get trainer
    model_trainer = Trainer(model, cfg)
    # `try-except` to save model before exiting if ^C was pressed
    try:
        # Start training + validation
        model_trainer.start()
    except KeyboardInterrupt or SystemExit:
        logger.info("Exit requested during train-val")
        # Collect state
        state = {
            "epoch": model_trainer.cfg.start_epoch,
            "best_loss": model_trainer.best_loss,
            "state_dict": model_trainer.net.state_dict(),
            "optimizer": model_trainer.optimizer.state_dict(),
        }
        logger.info("**** Saving state before exiting ****")
        # Save state if possible
        save_path = os.path.join(_HERE, "histovision",
                                 "checkpoints", cfg.final_weights_path)
        try:
            torch.save(state, save_path)
        except FileNotFoundError:
            # https://stackoverflow.com/a/273227
            Path(save_path).mkdir(parents=True, exist_ok=True)
        else:
            logger.info("Saved 🎉")
        # Exit
        sys.exit(0)

    # Helper function to plot scores
    def metric_plot(scores, name):
        plt.figure(figsize=(15, 5))
        # Plot training scores
        plt.plot(range(len(scores["train"])),
                 scores["train"],
                 label=f'train {name}')
        # Plot validation scores
        plt.plot(range(len(scores["val"])),
                 scores["val"],
                 label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    for metric_name, metric_values in model_trainer.meter.store.items():
        metric_plot(metric_values, metric_name)


if __name__ == "__main__":
    train()
