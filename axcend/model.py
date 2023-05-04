import torch
import torch.nn as nn
import numpy as np

from buildingblocks import Decoder, SingleConv, DoubleConv, ExtResNetBlock, \
    create_encoders, create_decoders

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='crg',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=32, layer_order='cge',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class CascadingUNet3D(nn.Module):
    """
    A sequence of 2 3D U-Nets.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=32, layer_order='cge',
                 num_groups=8, num_levels=[4, 3], is_segmentation=True, residual=True, conv_padding=1,
                 **kwargs):

        super(CascadingUNet3D, self).__init__()

        if residual:
            unet = ResidualUNet3D
        else:
            unet = UNet3D

        self.unet1 = unet(in_channels=in_channels,
                          out_channels=out_channels,
                          f_maps=f_maps,
                          layer_order=layer_order,
                          num_groups=num_groups,
                          num_levels=num_levels[0],
                          is_segmentation=False,
                          conv_padding=conv_padding,
                          **kwargs)

        self.unet2 = unet(in_channels=out_channels,
                          out_channels=out_channels,
                          final_sigmoid=final_sigmoid,
                          f_maps=f_maps,
                          layer_order=layer_order,
                          num_groups=num_groups,
                          num_levels=num_levels[1],
                          is_segmentation=is_segmentation,
                          conv_padding=conv_padding,
                          **kwargs)
        
        self.final_activation = self.unet2.final_activation
        self.testing = self.unet2.testing

    def forward(self, x):
        seg = self.unet1(x)
        cl = self.unet2(seg)
        if self.testing and self.final_activation:
            seg = self.final_activation(seg)
        return seg, cl


class AuxiliaryUNet3D(Abstract3DUNet):
    """
    3D U-Net model with an auxiliary classifier.
    """
    def __init__(self, in_channels, out_channels, out_channels_aux, input_size, final_sigmoid=True,
                 f_maps=32, layer_order='cge', conv_kernel_size=3, num_groups=8, num_levels=4,
                 is_segmentation=True, residual=True, conv_padding=1, **kwargs):

        if residual:
            basic_module = ExtResNetBlock
        else:
            basic_module = DoubleConv

        super(AuxiliaryUNet3D, self).__init__(in_channels=in_channels,
                                              out_channels=out_channels,
                                              final_sigmoid=final_sigmoid,
                                              basic_module=basic_module,
                                              conv_kernel_size=conv_kernel_size,
                                              f_maps=f_maps,
                                              layer_order=layer_order,
                                              num_groups=num_groups,
                                              num_levels=num_levels,
                                              is_segmentation=is_segmentation,
                                              conv_padding=conv_padding,
                                              **kwargs)

        in_channels_aux = self.encoders[-1].basic_module.out_channels

        if 'e' in layer_order:
            nonlinearity = nn.ELU(inplace=True)
        elif 'l' in layer_order:
            nonlinearity = nn.LeakyReLU(inplace=True)
        else:
            nonlinearity = nn.ReLU(inplace=True)

        conv = SingleConv(in_channels_aux, f_maps,
                          order=layer_order,
                          kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding)

        encoder_f_maps = 32*2**(num_levels-1)
        downsampled_size = np.array(input_size[-3:])//(2**(num_levels-1))
        pool_kernel_size = tuple(np.where(downsampled_size > 1, 2, 1))
        pool_kernel_size2 = tuple(np.where(downsampled_size > 4, 2, 1))
        in_channels_linear = encoder_f_maps*np.product(downsampled_size//pool_kernel_size//pool_kernel_size2)
        #in_channels_linear = f_maps*np.product(downsampled_size//pool_kernel_size)

        # TODO: Move aux classifier to its own class
        self.aux = nn.Sequential(conv,
                                 nn.AdaptiveAvgPool3d(1),
                                 nn.Flatten(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(f_maps, out_channels_aux))
        #self.aux = nn.Sequential(conv,
        #                         nn.MaxPool3d(kernel_size=pool_kernel_size),
        #                         nn.Flatten(),
        #                         nn.Linear(in_channels_linear, 1024),
        #                         nonlinearity,
        #                         nn.Dropout(p=0.5),
        #                         nn.Linear(1024, out_channels_aux))
        #self.aux = nn.Sequential(nn.MaxPool3d(kernel_size=pool_kernel_size),
        #                         nn.MaxPool3d(kernel_size=pool_kernel_size2),
        #                         nn.Flatten(),
        #                         nn.Linear(in_channels_linear, 1024),
        #                         nonlinearity,
        #                         nn.Dropout(p=0.5),
        #                         nn.Linear(1024, out_channels_aux))
        self.final_activation_aux = nn.Sigmoid() if out_channels_aux==1 else nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # compute auxiliary output
        y = self.aux(x)

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing:
            if self.final_activation is not None:
                x = self.final_activation(x)
            y = self.final_activation_aux(y)

        return x, y


class UNet2D(Abstract3DUNet):
    """
    Just a standard 2D Unet. Arises naturally by specifying conv_kernel_size=(1, 3, 3), pool_kernel_size=(1, 2, 2).
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)
