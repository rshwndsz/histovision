# Computational histopathology

## Installation

Start by cloning the repository
```bash
git clone https://github.com/rshwndsz/histovision
cd histovision
```
`histovision` uses the following directory structure
```bash
.
├── config
│  ├── hydra
│  ├── dataset
│  ├── model
│  └── config.yaml
├── histovision
│  ├── data
│  ├── datasets
│  ├── losses
│  ├── metrics
│  ├── models
│  ├── shared
│  ├── testers
│  └── trainers
├── outputs
├── README.md
├── environment.yml
├── test.py
└── train.py
```

## Getting started

### Datasets

Put in your datasets in `histovision/data` in the format shown below
```bash
histovision/data
├── carcinoma
│  ├── train
│  │  ├── imgs
│  │  └── masks
│  └── val
│     ├── imgs
│     └── masks
├── MoNuSeg
│  └── README.md
└── MoNuSeg_nitk
   ├── test
   │  ├── imgs
   │  └── masks
   ├── train
   │  ├── imgs
   │  └── masks
   └── val
      ├── imgs
      └── masks
```
Note that corresponding image and mask pairs must have the same name, i.e. the mask for `imgs/001.png` is `masks/001.png`.  
Next add dataset configuration in a config file inside `histovision/config/dataset`. 

### Models

Models go in `histovision/models`.  
The configuration for these models go in `histovision/config/model`.

### Losses

Loss functions or criterions go in `histovision/losses`.
Loss functions can be chosen by adding their full import to `histovision/config/config.yaml`.

### Trainers & Testers

`histovision` already comes with a trainer for segmentation tasks.  
Extend this trainer or create a new one by extending the base trainer `histovision.trainer.BaseTrainer`.
The same holds for testing too.

### Training & Testing

`histovision`'s configuration system is powered by [`facebookresearch/hydra`](https://hydra.cc).  
Training models can be as simple as 
```bash
python train.py \
model=unet dataset=MoNuSeg_nitk \
training.num_epochs=10 \ 
hyperparams.lr=1e-4
```

Similarly testing can be achieved using
```bash
python test.py \
model=unet dataset=MoNuSeg_nitk \
hydra.run.dir=outputs/02-02-02/20-00-00/
```

### Help 

For application help use
```bash
python train.py --help
```

To check the configuration file from the command line use
```bash
python train.py --hydra-help
```

## Credits

This project uses 
- [pytorch/pytorch](https://github.com/pytorch/pytorch)
- [opencv/opencv](https://github.com/opencv/opencv)
- [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations)
- [qubvel/segmentation-models.pytorch](https://https://github.com/qubvel/segmentation_models.pytorch)
- [facebookresearch/hydra](https://hydra.cc)
