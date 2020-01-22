from segmentation_models_pytorch import Unet

model = Unet('resnet34',
             classes=2,
             encoder_depth=4)
