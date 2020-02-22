import torch
import torch.nn as nn


class _ConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        Block performing Conv followed by BN, ReLU

        The output feature map size remains the same as
        that of input.
        :param in_ch: Number of input channels
        :param kernel_size: Size of filter
        """
        super(_ConvBNReluBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        return x


class _UpconvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        Block performing UpConv followed by BN, ReLU

        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param kernel_size: Size of filter
        """
        super(_UpconvBNReluBlock, self).__init__()
        self.layers = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch,
                                                       kernel_size),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU())

    def forward(self, x):
        x = self.layers(x)
        return x


class _DoubleConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        2 Blocks of successive Conv-BN-ReLU

        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param kernel_size: Size of filter
        """
        super(_DoubleConvBNReluBlock, self).__init__()
        self.block1 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block2 = _ConvBNReluBlock(out_ch, out_ch, kernel_size)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        return out1, out2


class _TripleConvBNReluBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        """
        3 block of Conv-BN-ReLU returning middle, final outputs

        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param kernel_size: Size of filter
        """
        super(_TripleConvBNReluBlock, self).__init__()
        self.block1 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block2 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)
        self.block3 = _ConvBNReluBlock(in_ch, out_ch, kernel_size)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        return out2, out3


class _PyramidPoolingBlock(nn.Module):
    def __init__(self, kernel_sizes, stride):
        """
        Stack of average-pooling layers returning concatenated tensor

        :param kernel_sizes: Size of the kernels in order of blocks
        :param stride: Stride value for convolution
        """
        super(_PyramidPoolingBlock, self).__init__()
        self.layers = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.layers.append(nn.AvgPool2d(kernel_size=kernel_size,
                                            stride=stride,
                                            padding='same'))

    def forward(self, x):
        out = []
        out[0] = self.layers[0](x)
        for i, layer in enumerate(self.layers[1:]):
            out[i] = layer(out[i-1])
        return torch.cat(out, 1)


class UnetPPL(nn.Module):
    def __init__(self, classes):
        """
        UNet with Pyramid Pooling Layers(UNetPPL)

        From https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8472806&tag=1
        :param classes: Number of classes
        """
        super(UnetPPL, self).__init__()
        self.input = nn.Sequential(nn.Conv2d(3, 64, 3), nn.ReLU())

        self.enc1 = _ConvBNReluBlock(64, 64, 3)
        self.enc2 = _TripleConvBNReluBlock(64, 128, 3)
        self.enc3 = _TripleConvBNReluBlock(128, 256, 3)
        self.enc4 = _TripleConvBNReluBlock(256, 512, 3)
        self.enc5 = _TripleConvBNReluBlock(512, 1024, 3)

        self.max_pool = nn.MaxPool2d(2, 2)

        self.ppl = _PyramidPoolingBlock(kernel_sizes=[1, 3, 8, 9, 11, 12, 13, 14],
                                        stride=2)

        self.dec1 = _UpconvBNReluBlock(1024*8, 1024, 3)
        self.dec2 = _DoubleConvBNReluBlock(2048, 1024, 3)
        self.dec3 = _DoubleConvBNReluBlock(1536, 768, 3)
        self.dec4 = _DoubleConvBNReluBlock(1024, 512, 3)
        self.dec5 = _DoubleConvBNReluBlock(768, 256, 3)
        self.dec6 = _DoubleConvBNReluBlock(320, classes, 3)

    def forward(self, x):
        out_input = self.input(x)

        skip1, out_enc1 = self.enc1(out_input)
        out_enc1 = self.max_pool(out_enc1)

        skip2, out_enc2 = self.enc2(out_enc1)
        out_enc2 = self.max_pool(out_enc2)

        skip3, out_enc3 = self.enc3(out_enc2)
        out_enc3 = self.max_pool(out_enc3)

        skip4, out_enc4 = self.enc4(out_enc3)
        out_enc4 = self.max_pool(out_enc4)

        skip5, out_enc5 = self.enc5(out_enc4)

        out_ppl = self.ppl(out_enc5)

        out_dec6 = self.dec6(out_ppl)

        out_dec5 = self.dec5(torch.cat([out_dec6, skip5], 1))

        out_dec4 = self.dec4(torch.cat([out_dec5, skip4], 1))

        out_dec3 = self.dec3(torch.cat([out_dec4, skip3], 1))

        out_dec2 = self.dec2(torch.cat([out_dec3, skip2], 1))

        out_dec1 = self.dec1(torch.cat([out_dec2, skip1], 1))

        return out_dec1
