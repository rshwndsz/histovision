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
# TODO Use loss based on config
from histovision.shared.loss import MixedLoss
# TODO Use dataset based on config
from histovision.datasets.MoNuSeg_nitk.api import provider
from histovision.datasets.MoNuSeg_nitk.api import DATA_FOLDER


class BinaryTrainer(object):
    """An object to encompass all training and validation

    Training loop, validation loop, logging, checkpoints are all
    implemented here.

    Attributes
    ----------
    cfg.val_freq : int
        Validation frequency
    device : torch.device
        GPU or CPU
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
        logger = logging.getLogger('root')
        self.cfg = cfg

        # Torch-specific initializations
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            self.device = torch.device("cuda:0")  # <<< Note: Single-GPU
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        logger.info(f"Using device {self.device}")

        # TODO Move to config
        # Model, loss, optimizer & scheduler
        self.net = model
        self.net = self.net.to(self.device)
        self.criterion = MixedLoss(9.0, 4.0)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.cfg.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=3, verbose=True,
                                           cooldown=0, min_lr=3e-6)

        # Get loaders for training and validation
        self.dataloaders = {
            phase: provider(
                root=DATA_FOLDER,
                phase=phase,
                batch_size=self.cfg.batch_size[phase],
                num_workers=self.cfg.num_workers,
                args={
                    'image_size': self.cfg.image_size,
                    'in_channels': self.cfg.in_channels
                }
            )
            for phase in self.cfg.phases
        }

        # Initialize losses & scores
        self.best_loss = float("inf")
        self.meter = Meter(self.cfg.phases, scores=self.cfg.scores)

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
        images = images.to(self.device)
        masks = targets.to(self.device)
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
        # TODO Add event system
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
        for epoch in range(1, self.cfg.num_epochs + 1):
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
                    logger = logging.getLogger('root')
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
