# Python STL
from pathlib import Path
import logging
# PyTorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
# Progress bars
from tqdm import tqdm
# Local
from histovision.shared.storage import Meter
# TODO Read loss from config
from histovision.losses import MixedLoss
# TODO Read dataset from config
from histovision.datasets.MoNuSeg_nitk.api import provider

# Get root logger
logger = logging.getLogger('root')


class BinaryTrainer(object):
    """An object to encompass all training and validation

    Training loop, validation loop, logging, checkpoints are all
    implemented here.

    Attributes
    ----------
    net
        Our NN in PyTorch
    criterion
        Loss function
    optimizer
        Optimizer
    scheduler
        Learning rate scheduler
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dataloaders for each phase
    best_loss : float
        Best validation loss
    meter : Meter
        Object to store loss & scores
    """
    def __init__(self, model, cfg):
        """Initialize a Trainer object

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model of your NN
        cfg : :obj:
            CLI arguments
        """
        # Save config
        self.cfg = cfg

        # TODO Read model from config
        # Model, loss, optimizer & scheduler
        self.net = model
        self.net = self.net.to(self.cfg.device)
        self.criterion = MixedLoss(9.0, 4.0)
        # TODO Read optimizer from config
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.cfg.hyperparams.lr)
        # TODO Read scheduler from config
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=3, verbose=True,
                                           cooldown=0, min_lr=3e-6)

        # Get loaders for training and validation
        self.dataloaders = {
            phase: provider(
                phase=phase,
                cfg=cfg
            )
            for phase in ('train', 'val')
        }

        # Initialize losses & scores
        self.best_loss = float("inf")
        self.meter = Meter(scores=self.cfg.scores)

    def forward(self, images, targets):
        """Forward pass

        Parameters
        ----------
        images : torch.Tensor
            Input to the NN
        targets : torch.Tensor
            Supervised labels for the NN

        Returns
        -------
        loss: torch.Tensor
            Loss from one forward pass
        logits: torch.Tensor
            Raw output of the NN, without any activation function
            in the last layer
        """
        images = images.to(self.cfg.device)
        masks = targets.to(self.cfg.device)
        logits = self.net(images)
        loss = self.criterion(logits, masks)
        return loss, logits

    def iterate(self, epoch, phase):
        """1 epoch in the life of a model

        Parameters
        ----------
        epoch : int
            Current epoch
        phase : str
            Phase of learning
            In ['train', 'val']
        Returns
        -------
        epoch_loss: float
            Average loss for the epoch
        """
        # Set model & dataloader based on phase
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        # ===ON_EPOCH_BEGIN===
        self.meter.on_epoch_begin(epoch, phase)

        # Learning!
        self.optimizer.zero_grad()

        for itr, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Load images and targets
            images, targets = batch

            # ===ON_BATCH_BEGIN===
            self.meter.on_batch_begin()

            # Forward pass
            loss, logits = self.forward(images, targets)
            if phase == "train":
                # Backprop for training only
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach().cpu()
                logits = logits.detach().cpu()

                # ===ON_BATCH_CLOSE===
                self.meter.on_batch_close(loss=loss,
                                          logits=logits, targets=targets)

        # ===ON_EPOCH_CLOSE===
        # Collect loss & scores
        self.meter.on_epoch_close()

        # Empty GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return average loss from the criterion for this epoch
        return self.meter.store['loss'][phase][-1]

    def start(self):
        """Start the loops!"""

        # ===ON_TRAIN_BEGIN===
        self.meter.on_train_begin()

        # <<< Change: Hardcoded starting epoch
        for epoch in range(1, self.cfg.hyperparams.num_epochs + 1):
            # Update start_epoch
            self.cfg.start_epoch = epoch

            # Train model for 1 epoch
            self.iterate(epoch, "train")

            # Construct the state for a possible save later
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            # Validate model for `val_freq` epochs
            if epoch % self.cfg.val_freq == 0:
                val_loss = self.iterate(epoch, "val")

                # Step the scheduler based on validation loss
                self.scheduler.step(val_loss)

                # TODO Add EarlyStopping

                # Save model if val loss is lesser than anything seen before
                if val_loss < self.best_loss:
                    logger.info(f"**** New optimal found, saving state in "
                                f"{self.cfg.best_weights_path} ****")
                    state["best_loss"] = self.best_loss = val_loss
                    Path(self.cfg.best_weights_path).mkdir(
                        parents=True, exist_ok=True)
                    torch.save(state, self.cfg.best_weights_path)

            # Print newline after every epoch
            print()

        # ===ON_TRAIN_END===
        self.meter.on_train_close()
