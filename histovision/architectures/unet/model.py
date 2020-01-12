from segmentation_models_pytorch import Unet

model = smp.Unet('resnet34',
                 classes=2,
                 encoder_depth=4)

