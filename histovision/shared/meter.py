# Python STL
from datetime import datetime
import logging
# PyTorch
import torch
# Local
from histovision.shared import utils
import histovision.metrics as metrics

# Get root logger
logger = logging.getLogger('root')


class BaseMeter(object):
    def __init__(self):
        pass

    def on_train_begin(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_close(self, *args, **kwargs):
        pass

    def on_epoch_close(self, *args, **kwargs):
        pass

    def on_train_close(self, *args, **kwargs):
        pass


class AverageMeter(BaseMeter):
    """Object to log & hold values during training"""
    def __init__(self, scores, phases=('train', 'val')):
        super(AverageMeter, self).__init__()
        self.phases = phases
        self.current_phase = phases[0]
        self.scores = scores
        # Storage over all epochs
        self.store = {
            score: {
                phase: [] for phase in self.phases
            } for score in self.scores.keys()
        }
        self.base_threshold = 0.5
        # Storage over 1 single epoch
        self.metrics = {
            score: [] for score in self.scores
        }
        self.epoch_start_time = datetime.now()

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, current_epoch, current_phase):
        # Log epoch, phase and start time
        self.epoch_start_time = datetime.now()
        epoch_start_time_string = datetime.strftime(self.epoch_start_time,
                                                    '%I:%M:%S %p')
        logger.info(f"Starting epoch: {current_epoch} | "
                    f"phase: {current_phase} | "
                    f"@ {epoch_start_time_string}")

        # Initialize metrics
        self.metrics = {
            score: [] for score in self.scores
        }

        # For later
        self.current_phase = current_phase

    def on_batch_begin(self):
        pass

    def on_batch_close(self, loss, outputs, targets):
        # Get predictions and probabilities from raw logits
        preds = utils.predict(outputs, self.base_threshold)

        # Assertion for shapes
        if not (preds.shape == targets.shape):
            raise ValueError(f"Shape of outputs: {outputs.shape} must be the same "
                             f"as that of targets: {targets.shape}.")

        # Add loss to list
        self.metrics['loss'].append(loss)

        # Calculate and add to metric lists
        dice = metrics.dice_score(preds, targets, self.base_threshold)
        self.metrics['dice'].append(dice)

        iou = metrics.iou_score(preds, targets)
        self.metrics['iou'].append(iou)

        acc = metrics.accuracy_score(preds, targets)
        self.metrics['acc'].append(acc)

        # <<< Change: Hardcoded for binary segmentation
        prec = metrics.precision_score(preds, targets)[1]
        self.metrics['prec'].append(prec)

    def on_epoch_close(self):
        # Average over metrics obtained for every batch in the current epoch
        self.metrics.update({key: [
            utils.nanmean(torch.tensor(self.metrics[key])).item()
        ] for key in self.metrics.keys()})

        # Compute time taken to complete the epoch
        epoch_end_time = datetime.now()
        delta_t = epoch_end_time - self.epoch_start_time

        # Construct string for logging
        metric_string = f""
        for metric_name, metric_value in self.metrics.items():
            metric_string += f"{metric_name}: {metric_value[0]:.4f} | "
        metric_string += f"in {delta_t.seconds}s"

        # Log metrics & time taken
        logger.info(f"{metric_string}")

        # Put metrics for this epoch in long term (complete training) storage
        for s in self.store.keys():
            try:
                self.store[s][self.current_phase].extend(self.metrics[s])
            except KeyError:
                logger.warning(f"Key '{s}' not found. Skipping...",
                               exc_info=True)
                continue

    def on_train_close(self):
        pass
