# Python STL
from pathlib import Path
import logging
# Image Processing
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
            logger.info(f"Found {len(self.image_paths)} images in {_path_to_imgs}")

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
        # <<< Note:
        # Masks should have int values in [0, C-1] where C => Number of classes
        # Checking the above condition for every mask is inefficient
        # So it's left to you 🙇
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Check if mask has been read properly
        if mask.size == 0:
            raise IOError(f"Unable to load mask: {mask_path}")

        # Augment masks and images
        augmented = self.transforms['aug'](image=image, mask=mask)
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
        # Constants from args
        image_size = (cfg.dataset.image_size, cfg.dataset.image_size)
        # Transforms for both images & masks
        common_tfs = []

        if phase == "train":
            # Data augmentation for training only
            common_tfs.extend([
                tf.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5),
                tf.Flip(p=0.5),
                tf.RandomRotate90(p=0.5),
            ])
            # Exotic Augmentations for train only
            common_tfs.extend([
                tf.RandomBrightnessContrast(p=0.5),
                tf.ElasticTransform(p=0.5),
                tf.MultiplicativeNoise(multiplier=(0.5, 1.5),
                                       per_channel=True, p=0.2),
            ])
        # Crop all images & masks to provided size
        common_tfs.extend([
            tf.RandomSizedCrop(min_max_height=image_size,
                               height=image_size[0],
                               width=image_size[1],
                               w2h_ratio=1.0,
                               interpolation=cv2.INTER_LINEAR,
                               p=1.0),
        ])
        common_tfs = Compose(common_tfs)

        # Mask only transforms
        # TODO Replace by class dict mapping
        mask_tfs = Compose([
            tf.Normalize(mean=0, std=1, always_apply=True)
        ])
        # Image only transforms
        image_tfs = Compose([
            tf.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), always_apply=True)
        ])
        # Image to tensor
        final_tfs = Compose([
            ToTensorV2()
        ])

        transforms = {
            'aug': common_tfs,
            'img_only': image_tfs,
            'mask_only': mask_tfs,
            'final': final_tfs
        }
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
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle
    )

    return dataloader
