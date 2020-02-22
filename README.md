# Computational histopathology

An attempt to build a benchmark repository of state-of-the-art architectures & methods in segmentation & classification of histopathology images.

## Getting started

Training  
```bash
python train.py \
model=unet dataset=MoNuSeg_nitk \
training.num_epochs=10 \ 
hyperparams.lr=1e-4
```

Testing  
```bash
python test.py \
model=unet dataset=MoNuSeg_nitk \
hydra.run.dir=outputs/02-02-02/20-00-00/
```

Help
```bash
python train.py --help
```
