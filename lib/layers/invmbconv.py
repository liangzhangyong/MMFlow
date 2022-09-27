# ----------------------------------------------------- #
#    Written by zyliang (zyliang18@mails.jlu.edu.cn)    #
# ----------------------------------------------------- #

import numpy as np
import torch
import torch.nn as nn

from matrix_utils import power_series_matrix_logarithm_trace
from spectral_norm_conv import spectral_norm_conv
from spectral_norm_fc import spectral_norm_fc

from typing import Type, Callable, Tuple, Optional, Set, List, Union

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath


# ------------------------------------------------------------------------------------------ #
class MBConv(nn.Module):
    """
        Without downsampling: x ← x + Proj(SE(DWConv(Conv(Norm(x)))))
        With downsampling: x ← Proj(Pool2D(x)) + Proj(SE(DWConv ↓(Conv(Norm(x))))).
        Conv is a 1 X 1 convolution followed by a Batch Normalization layer and a GELU activation.
        SE is the Squeeze-Excitation layer.
        Proj is the shrink 1 X 1 convolution.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_shape: torch.Tensor,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConv, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Check parameters for downscaling
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            _wrapper_spectral_norm(
                DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=2 if downscale else 1,
                                       act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path), shapes=in_shape,
                kernel_size=3),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            _wrapper_spectral_norm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1)),
                                   shapes=in_shape, kernel_size=1)
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            _wrapper_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
                                   shapes=in_shape, kernel_size=1)
        ) if downscale else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


# -------------------------------------------------------------------------------------- #
class InvertibleMBConv(nn.Module):
    def __init__(
            self,
            in_shape: torch.Tensor,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            numTraceSamples: int = 1,
            numSeriesTerms: int = 5,
            drop_path: float = 0.,
    ) -> None:
        super(InvertibleMBConv, self).__init__()
        self.MBConv_branch = MBConv(in_shape, in_channels, out_channels, downscale, act_layer, norm_layer, drop_path)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms

    def forward(self, x: torch.Tensor, ignore_logdet: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        Fx = self.MBConv_branch(x)
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        x = x + Fx
        return x, trace

    def inverse(self, y: torch.Tensor, maxIter: int = 100) -> torch.Tensor:
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)  need to modify
            x = y
            for iter_index in range(maxIter):
                summand = self.MBConv_branch(x)
                x = y - summand
            return x


# -------------------------------------------------------------------------------------- #
def _gelu_ignore_parameters(*args, **kwargs) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.
    Args:
        *args: Ignored.
        **kwargs: Ignored.
    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


def _wrapper_spectral_norm(self, layer: nn.Module, shapes: torch.Tensor, kernel_size: int) -> nn.Module:
    if kernel_size == 1:
        # use spectral norm fc, because bound are tight for 1x1 convolutions
        return spectral_norm_fc(layer, coeff=0.97, n_power_iterations=5)
    else:
        # use spectral norm based on conv, because bound not tight
        return spectral_norm_conv(layer, coeff=0.97, input_dim=shapes, n_power_iterations=5)


###################################################################
if __name__ == '__main__':
    def test_MBConv() -> None:
        input = torch.rand(1, 128, 28, 28)
        mbconv = MBConv(in_shape=input.shape, in_channels=128, out_channels=256, downscale=True)
        output = mbconv(input)
        print('the shape of MBConv:', output.shape)


    def test_InvMBConv() -> None:
        input = torch.rand(1, 128, 28, 28).requires_grad_()
        inv_mbconv = InvertibleMBConv(in_shape=input.shape, in_channels=128, out_channels=256, downscale=True)
        output_forword = inv_mbconv.forward(input)
        print('the shape of Invertible MBConv for forward:', output_forword[0].shape)
        output_inverse = inv_mbconv.inverse(input)
        print('the shape of Invertible MBConv for inverse:', output_inverse.shape)


    # UnitTest
    test_MBConv()
    test_InvMBConv()
