# Python STL
from pathlib import Path
import logging
# For original cwd
from hydra.utils import get_original_cwd
# Image Processing
import numpy as np
import cv2
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
# Data augmentation
from albumentations.augmentations import transforms as tf
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Get root logger
logger = logging.getLogger('root')


class SegmentationDataset(Dataset):
    def __init__(self, phase, cfg):
        """Create an API for the dataset

        Parameters
        ----------
        phase : str
            Phase of learning
            In ['train', 'val']
        cfg
            User specified configuration
        """
        # Save config
        self.cfg = cfg
        self.phase = phase
        logger.info(f"Phase: {phase}")

        # Data Augmentations and tensor transformations
        self.transforms = SegmentationDataset.get_transforms(self.phase, self.cfg)

        # Get absolute paths of all images in {root}/{phase}/imgs
        _path_to_imgs = Path(self.cfg.dataset.root) / self.phase / "imgs"
        self.image_paths = sorted(list(_path_to_imgs.glob(self.cfg.dataset.image_glob)))

        # Check if all images have been read properly
        if len(self.image_paths) == 0:
            raise IOError(f"No images found in {_path_to_imgs}")
        else:
            logger.info(f"Found {len(self.image_paths)} images in {_path_to_imgs.relative_to(get_original_cwd())}")

    def __getitem__(self, idx):
        # Get path to image
        image_path = self.image_paths[idx]
        # Read image with opencv
        if self.cfg.dataset.in_channels == 1:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(image_path))
        # Check if image has been read properly
        if image.size == 0:
            raise IOError(f"Unable to load image: {image_path}")

        # <<< Note:
        # Mask is supposed to have the same filename as image
        mask_path = Path(self.cfg.dataset.root) / self.phase / "masks" / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Check if mask has been read properly
        if mask.size == 0:
            raise IOError(f"Unable to load mask: {mask_path}")

        # Map pixel intensities to classes
        for intensity, kclass in self.cfg.dataset.class_dict.items():
            mask = np.where(mask == int(intensity), float(kclass), mask)
        mask = mask.astype(np.float32)

        # Augment masks and images
        augmented = self.transforms['common'](image=image, mask=mask)
        new_image = self.transforms['img_only'](image=augmented['image'])
        new_mask = self.transforms['mask_only'](image=augmented['mask'])
        aug_tensors = self.transforms['final'](image=new_image['image'], mask=new_mask['image'])
        image = aug_tensors['image']
        mask = aug_tensors['mask']

        # Add a channel dimension (C in [N C H W] in PyTorch) if required
        if self.cfg.dataset.num_classes == 2:
            mask = torch.unsqueeze(mask, dim=0)  # [H, W] => [H, C, W]

        # Return tuple of tensors
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    # TODO Read transforms from config
    @staticmethod
    def get_transforms(phase, cfg):
        """Get composed albumentations augmentations

        Parameters
        ----------
        phase : str
            Phase of learning
            In ['train', 'val']
        cfg
            User specified configuration

        Returns
        -------
        transforms: dict[str, albumentations.core.composition.Compose]
            Composed list of transforms
        """
        transforms = {
            'common': [],       # Common for both image & mask
            'img_only': [],     # Image only
            'mask_only': [],    # Mask only
            'final': []         # Tfs applied after previous 3, Must include ToTensorV2
        }
        # Collect transforms from config file
        for tf_type in cfg.augmentations[phase].keys():
            if cfg.augmentations[phase][tf_type] is None:
                continue
            for aug, params in cfg.augmentations[phase][tf_type].items():
                if aug == 'ToTensorV2':
                    transforms[tf_type].append(ToTensorV2())
                elif params is None:
                    transforms[tf_type].append(tf.__dict__[aug]())
                else:
                    transforms[tf_type].append(tf.__dict__[aug](**params))

        # Compose transforms
        transforms = dict((k, Compose(v)) for k, v in transforms.items())

        return transforms


def provider(phase, cfg):
    """Return dataloader for a given dataset & phase

    Parameters
    ----------
    phase : str
        Phase of learning
        In ['train', 'val']
    cfg
        User specified configuration

    Returns
    -------
    dataloader: DataLoader
        DataLoader for loading data from CPU to GPU
    """
    # Get dataset
    logger.info(f"Collecting {phase} dataset")
    image_dataset = SegmentationDataset(phase, cfg)

    # Get dataloader
    logger.info(f"Creating {phase} dataloader")
    dataloader = DataLoader(
        image_dataset,
        batch_size=cfg.hyperparams.batch_size[phase],
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        shuffle=cfg.dataloader.shuffle
    )

    return dataloader
