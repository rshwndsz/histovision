# Python STL
from datetime import datetime
import logging
# PyTorch
import torch
# Config
import hydra

# Get root logger
logger = logging.getLogger('root')

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


class AverageMeter(object):
    """Object to log & hold values during training"""
    def __init__(self, scores, mode, from_logits=True, include_classes=None, phases=('train', 'val')):
        super(AverageMeter, self).__init__()
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        self.mode = mode
        self.from_logits = from_logits
        self.include_classes = include_classes

        self.phases = phases
        self.current_phase = phases[0]
        self.scores = scores
        # Storage over all epochs
        self.store = {
            score_name: {
                phase: [] for phase in self.phases
            } for score_name in ["loss", *self.scores.keys()]
        }
        self.base_threshold = 0.5
        # Storage over 1 single epoch
        self.metrics = {
            score_name: [] for score_name in ["loss", *self.scores.keys()]
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
            score: [] for score in ["loss", *self.scores.keys()]
        }

        # For later
        self.current_phase = current_phase

    def on_batch_begin(self):
        pass

    def on_batch_close(self, loss, outputs, targets):
        assert outputs.size(0) == targets.size(0), "Batch size must be same"
        # Add loss to list
        self.metrics['loss'].append(loss)
        # Add other metrics to list
        for score_name, score_method_name in self.scores.items():
            fn = hydra.utils.get_method(score_method_name)
            score = fn(outputs, targets,
                       mode=self.mode, from_logits=self.from_logits,
                       include_classes=self.include_classes)
            self.metrics[score_name].append(score)

    def on_epoch_close(self):
        # Average over metrics obtained for every batch in the current epoch
        self.metrics.update({key: torch.mean(torch.stack(self.metrics[key], dim=0), dim=0)
                             for key in self.metrics.keys()})

        # Compute time taken to complete the epoch
        epoch_end_time = datetime.now()
        delta_t = epoch_end_time - self.epoch_start_time

        # Construct string for logging
        metric_string = f""
        for metric_name, metric_value in self.metrics.items():
            metric_string += f"{metric_name}: {metric_value.numpy()} | "
        metric_string += f"in {delta_t.seconds}s"

        # Log metrics & time taken
        logger.info(f"{metric_string}")

        # Put metrics for this epoch in long term storage
        for s in self.store.keys():
            try:
                self.store[s][self.current_phase].append(self.metrics[s])
            except KeyError:
                logger.warning(f"Key '{s}' not found. Skipping...", exc_info=True)
                continue

    def on_train_close(self):
        pass
