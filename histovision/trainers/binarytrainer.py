# Python STL
from pathlib import Path
import logging
# PyTorch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# Progress bars
from tqdm import tqdm
# To get class, method from string
import hydra.utils
# Local
from histovision.trainers import BaseTrainer
from histovision.shared.meter import AverageMeter

# Get root logger
logger = logging.getLogger('root')


class BinaryTrainer(BaseTrainer):
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
    meter : AverageMeter
        Object to store loss & scores
    """
    def __init__(self, cfg):
        """Initialize a Trainer object

        Parameters
        ----------
        cfg : :obj:
            CLI arguments
        """
        super(BinaryTrainer, self).__init__()
        # Save config
        self.cfg = cfg

        # Model, loss, optimizer & scheduler
        self.net = hydra.utils.instantiate(self.cfg.model).to(self.cfg.device)
        self.criterion = hydra.utils.instantiate(self.cfg.criterion)
        self.optimizer = optim.__dict__[self.cfg.optimizer](self.net.parameters(), lr=self.cfg.hyperparams.lr)
        self.scheduler = lr_scheduler.__dict__[self.cfg.scheduler['class']](self.optimizer, **self.cfg.scheduler.params)

        # Get loaders for training and validation
        self.dataloaders = {
            phase: hydra.utils.get_class(cfg.provider)(
                phase=phase,
                cfg=cfg
            )
            for phase in ('train', 'val')
        }

        # current_epoch that can be accessed while saving
        self.current_epoch = cfg.training.start_epoch

        # Initialize losses & scores
        self.best_loss = float("inf")
        self.meter = AverageMeter(scores=self.cfg.scores, mode="multiclass",
                                  from_logits=True, include_classes=None)

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
        outputs: torch.Tensor
            Raw output of the NN, without any activation function
            in the last layer
        """
        images = images.to(self.cfg.device)
        if self.cfg.criterion['name'] == "cross_entropy":
            targets = targets.long()
        masks = targets.to(self.cfg.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

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
            loss, outputs = self.forward(images, targets)
            if phase == "train":
                # Backprop for training only
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach().cpu()
                outputs = outputs.detach().cpu()

                # ===ON_BATCH_CLOSE===
                self.meter.on_batch_close(loss=loss, outputs=outputs, targets=targets)

        # ===ON_EPOCH_CLOSE===
        # Collect loss & scores
        self.meter.on_epoch_close()

        # Empty GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return average loss from the criterion for this epoch
        return self.meter.store['loss'][phase][-1]

    def validate(self, epoch):
        # Go through 1 validation epoch & get validation loss
        val_loss = self.iterate(epoch, "val")
        # TODO Display validation predictions One of ("every", "best", "every n val")
        # Step the scheduler based on validation loss
        self.scheduler.step(val_loss)
        # Save model if val loss is lesser than anything seen before
        if val_loss < self.best_loss:
            # Construct the state dict
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            logger.info(f"**** New optimal found, saving state in "
                        f"{Path(self.cfg.best_weights_path)} ****")
            state["best_loss"] = self.best_loss = val_loss
            Path(self.cfg.best_weights_path).parent.mkdir(
                parents=True, exist_ok=True)
            try:
                torch.save(state, self.cfg.best_weights_path)
            except IOError:
                logger.warning(f"Could not save in {self.cfg.best_weights_path}", exc_info=True)
            else:
                logger.info("Saved 🎉")

    def start(self):
        """Start the loops!"""
        # ===ON_TRAIN_BEGIN===
        self.meter.on_train_begin()

        # Train for `num_epochs` from `start_epoch`
        for epoch in range(self.cfg.training.start_epoch, self.cfg.training.num_epochs + 1):
            self.current_epoch = epoch
            # Train model for 1 epoch
            self.iterate(epoch, "train")
            # Validate model after `val_freq` epochs
            if epoch % self.cfg.training.val_freq == 0:
                self.validate(epoch)

            # Print newline after every epoch
            print()

        # ===ON_TRAIN_END===
        self.meter.on_train_close()
