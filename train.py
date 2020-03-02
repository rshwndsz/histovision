# Imports
# Python STL
import logging
from pathlib import Path
import sys
# Plotting
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.backends.cudnn as cudnn
# Advanced configurations
import hydra

# Get root logger
logger = logging.getLogger('root')


@hydra.main(config_path="config/config.yaml")
def train(cfg):
    # Show working directory
    logger.info(f"Running experiment in {Path.cwd()}")
    # Validate configuration
    cfg = validate_config(cfg)

    # Set constants based on config
    # Faster convolutions at the expense of memory
    cudnn.benchmark = cfg.training.cudnn_benchmark

    # Get trainer from config
    trainer = hydra.utils.get_class(cfg.trainer)(cfg)

    # `try-except` to save model before exiting if ^C was pressed
    try:
        # Start training + validation
        trainer.start()
    except KeyboardInterrupt or SystemExit:
        logger.info("Exit requested during train-val")
        # Don't save state in debugging mode
        if cfg.debugging:
            logger.warning("NOT saving state in DEBUG mode")
            sys.exit(0)
        # Collect state
        state = {
            "epoch": trainer.current_epoch,
            "best_loss": trainer.best_loss,
            "state_dict": trainer.net.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
        }
        logger.info("**** Saving state before exiting ****")
        # Create file with parent directories if it doesn't exist
        Path(cfg.final_weights_path).parent.mkdir(parents=True, exist_ok=True)
        # Save state
        try:
            torch.save(state, cfg.final_weights_path)
        except IOError:
            logger.warning(f"Could not save in {cfg.final_weights_path}", exc_info=True)
        else:
            logger.info("Saved 🎉")
        # Exit
        sys.exit(0)

    for metric_name, metric_values in trainer.meter.store.items():
        metric_plot(cfg, metric_values, metric_name)


# TODO Validate everything else in config
def validate_config(cfg):
    # dataset
    # dataset.root
    if not Path(cfg.dataset.root).is_dir():
        raise NotADirectoryError(f"{cfg.dataset.root} is not a directory or it doesn't exist.")

    # dataset.root/imgs/[train, val]/[imgs, masks]
    _path_to_imgs = Path(cfg.dataset.root) / "train" / "imgs"
    if not Path(_path_to_imgs).is_dir():
        raise FileNotFoundError(f"{_path_to_imgs} doesn't exist.")
    _path_to_imgs = Path(cfg.dataset.root) / "val" / "imgs"
    if not Path(_path_to_imgs).is_dir():
        raise FileNotFoundError(f"{_path_to_imgs} doesn't exist.")
    _path_to_imgs = Path(cfg.dataset.root) / "train" / "masks"
    if not Path(_path_to_imgs).is_dir():
        raise FileNotFoundError(f"{_path_to_imgs} doesn't exist.")
    _path_to_imgs = Path(cfg.dataset.root) / "val" / "masks"
    if not Path(_path_to_imgs).is_dir():
        raise FileNotFoundError(f"{_path_to_imgs} doesn't exist.")

    # dataset.num_classes
    if not cfg.dataset.num_classes >= 2:
        raise ValueError(f"Number of classes must be >= 2. 2 => Binary, >2 => Multi-class")

    # dataset.class_dict
    if not len(cfg.dataset.class_dict) == cfg.dataset.num_classes:
        raise ValueError(f"Length of class dict must be same as number of classes.")

    # device
    if torch.cuda.is_available() and cfg.device == 'cpu':
        logger.warning("Using device: 'cpu' when device: 'cuda' is available")

    elif not torch.cuda.is_available() and cfg.device != 'cpu':
        logger.warning("Setting device to 'cpu' as device: 'cuda' is not available")
        cfg.device = 'cpu'

    return cfg


# Helper function to plot scores at the end of training
# TODO Fix
def metric_plot(cfg, scores, name):
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
    # Create path if it doesn't exist
    save_path = Path(cfg.training.results_dir) / f'{name}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=1)


if __name__ == "__main__":
    train()
