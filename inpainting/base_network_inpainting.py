# Copyright (c) 2019 Lightricks. All rights reserved.

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class Upsample(nn.Module):
    """
    Up sampling module
    """

    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class HardSigmoid(torch.nn.Module):
    """
    Pytorch implementation of the hard sigmoid activation function
    """

    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, input):
        x = (0.2 * input) + 0.5
        x = torch.clamp(x, 0, 1)
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x


class GatedConv2d(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        activation=torch.nn.ELU(1.0, inplace=True),
        dropout=0,
        gate_type="regular_conv",
    ):
        super(GatedConv2d, self).__init__()
        self.stride = stride

        padding = (dilation * (kernel_size - 1)) // 2
        # assert stride <= 2  # Pytorch behaves weirdly when the stride is > 1, and I can handle only when its <= 2
        # if stride == 2:
        #     padding += 1

        self.activation = activation
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        if gate_type == "regular_conv":
            self.mask_conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        elif gate_type == "single_channel":
            self.mask_conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        elif gate_type == "pixel_wise":
            self.mask_conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=dilation,
                bias=bias,
            )
        elif gate_type == "depth_separable":
            self.mask_conv2d = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    groups=in_channels,
                ),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=bias,
                ),
            )

        self.sigmoid = nn.Sigmoid()
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        # to deal with the weird stride behaviour. I can remove this if I'll retrain the nn
        # if self.stride == 2:
        #     x = x[:,:,1:,1:]
        #     mask = mask[:,:,1:,1:]
        x = x * self.sigmoid(mask)

        if self.activation is not None:
            x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class GatedGeneratorSubNetwork(torch.nn.Module):
    """
    One of the 2 subnetworks of the inpainting generator. Used both for the coarse subnetwork and the refined one
    """

    def __init__(
        self,
        inference=True,
        n_in_channel=4,
        depth_factor=32,
        masking=True,
        dropout=0,
        activation="elu",
        gate_type="regular_conv",
        kernel_size=3
    ):
        super(GatedGeneratorSubNetwork, self).__init__()
        self.inference = inference
        self.masking = masking
        self.gate_type = gate_type
        gated_activation = self._activations_map(activation)

        modules = []
        # The first convolution is not gated for some reason
        modules.append(
            nn.Conv2d(in_channels=n_in_channel, out_channels=depth_factor, kernel_size=5, padding=2)
        )
        modules.append(gated_activation())
        # now we have 5 gated convs
        modules.append(
            GatedConv2d(
                in_channels=depth_factor,
                out_channels=depth_factor * 2,
                kernel_size=kernel_size,
                stride=2,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 2,
                out_channels=depth_factor * 2,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )

        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 2,
                out_channels=depth_factor * 4,
                kernel_size=kernel_size,
                stride=2,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )

        # now we have 4 dialated convolutions
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                dilation=2,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                dilation=4,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                dilation=8,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                dilation=16,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )

        # 2 more gated convs
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 4,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )

        if dropout > 0:
            modules.append(nn.Dropout(dropout))

        # upsampling and gated convs
        modules.append(Upsample(scale_factor=2))
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 2,
                kernel_size=3,
                dropout=dropout,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 2,
                out_channels=depth_factor * 2,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(Upsample(scale_factor=2))
        modules.append(
            GatedConv2d(
                in_channels=depth_factor * 2,
                out_channels=depth_factor,
                kernel_size=3,
                dropout=dropout,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor,
                out_channels=depth_factor // 2,
                kernel_size=3,
                activation=gated_activation(),
                gate_type=gate_type,
            )
        )
        modules.append(
            GatedConv2d(
                in_channels=depth_factor // 2,
                out_channels=3,
                kernel_size=3,
                activation=None,
                gate_type=gate_type,
            )
        )

        self.gated_subnetwork = nn.Sequential(*modules)
        self.hard_sigmoid = HardSigmoid()

    def forward(self, image, mask):
        mask3 = torch.cat([mask, mask, mask], dim=1)
        if self.masking:
            masked_img = image * (1 - mask3)
        else:
            masked_img = image
        x = torch.cat([masked_img, mask], dim=1)
        for i, module in enumerate(self.gated_subnetwork):
            x = module(x)

        if self.inference:
            x = 2.5 * x
            x = self.hard_sigmoid(x)
            x = 2.0 * x - 1
        else:
            x = torch.clamp(x, -1.0, 1.0)

        return x

    def _activations_map(self, activation):
        if activation == "elu":
            return lambda: nn.ELU(1.0, inplace=True)
        elif activation == "leaky_relu":
            return lambda: nn.LeakyReLU(negative_slope=0.2)
        elif activation == "tanh":
            return nn.Tanh
        elif activation == "sigmoid":
            return nn.Sigmoid
        elif activation == "relu":
            return nn.ReLU
        else:
            raise NotImplementedError()


class InpaintingGenerator(nn.Module):
    """
    The generator network of the inpainting model
    """

    def __init__(
        self,
        depth_factor=32,
        inference=True,
        activation="elu",
        gate_type_coarse="regular_conv",
        gate_type_fine="regular_conv",
        kernel_size=3
    ):
        super(InpaintingGenerator, self).__init__()
        self.inference = inference
        self.coarse_subnetwork = GatedGeneratorSubNetwork(
            depth_factor=depth_factor,
            inference=inference,
            masking=True,
            activation=activation,
            gate_type=gate_type_coarse,
            kernel_size=kernel_size
        )
        self.refined_subnetwork = GatedGeneratorSubNetwork(
            depth_factor=depth_factor,
            inference=inference,
            masking=False,
            activation=activation,
            gate_type=gate_type_fine,
            kernel_size=kernel_size
        )

    def forward(self, image, mask):
        # image = 2 * image - 1  // this is not needed now as we do it when we generate the image
        coarse_out = self.coarse_subnetwork(image, mask)
        inner = coarse_out * mask
        flipped_mask = 1 - mask
        masked_img = image * flipped_mask
        combined = masked_img + inner
        fine_out = self.refined_subnetwork(combined, mask)
        inner_fine = fine_out * mask
        out = masked_img + inner_fine
        # out = 0.5 * out + 0.5
        return out, fine_out, coarse_out


class InpaintingDiscriminator(nn.Module):
    """
    The discriminator network of the inpainting model
    """

    def __init__(self, depth_factor=32, out_activation=None):
        super(InpaintingDiscriminator, self).__init__()

        if not out_activation:
            self.out_activation = nn.Identity()
        else:
            self.out_activation = out_activation

        modules = []
        modules.append(
            nn.Conv2d(in_channels=4, out_channels=depth_factor * 2, kernel_size=5, padding=2)
        )
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        modules.append(
            nn.Conv2d(
                in_channels=depth_factor * 2,
                out_channels=depth_factor * 4,
                kernel_size=5,
                padding=2,
                stride=2,
            )
        )
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        modules.append(
            nn.Conv2d(
                in_channels=depth_factor * 4,
                out_channels=depth_factor * 8,
                kernel_size=5,
                padding=2,
                stride=2,
            )
        )
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        modules.append(
            nn.Conv2d(
                in_channels=depth_factor * 8,
                out_channels=depth_factor * 8,
                kernel_size=5,
                padding=2,
                stride=2,
            )
        )
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        modules.append(
            nn.Conv2d(
                in_channels=depth_factor * 8,
                out_channels=depth_factor * 8,
                kernel_size=5,
                padding=2,
                stride=2,
            )
        )
        modules.append(nn.LeakyReLU(negative_slope=0.2))
        modules.append(
            nn.Conv2d(
                in_channels=depth_factor * 8,
                out_channels=depth_factor * 8,
                kernel_size=5,
                padding=2,
                stride=2,
            )
        )
        modules.append(self.out_activation)
        self.convolutional_model = nn.Sequential(*modules)

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        x = self.convolutional_model(x)
        x = x.view((x.size(0), -1))  # flatten the output
        return x
