# ----------------------------------------------------- #
#    Written by zyliang (zyliang18@mails.jlu.edu.cn)    #
# ----------------------------------------------------- #

import torch
import torch.nn as nn

from matrix_utils import power_series_matrix_logarithm_trace
from spectral_norm_fc import spectral_norm_fc

from typing import Type, Callable, Tuple

from timm.models.layers import DropPath

from invattention import RelativeSelfAttention
from mlp import Mlp


# ------------------------------------------------------------------------------------------ #
class GBTrans(nn.Module):
    """ MaxViT Transformer block.
        With block partition:
        x ← x + Unblock(RelAttention(Block(LN(x))))
        x ← x + MLP(LN(x))

        With grid partition:
        x ← x + Ungrid(RelAttention(Grid(LN(x))))
        x ← x + MLP(LN(x))

        Layer Normalization (LN) is applied after the grid/window partition to prevent multiple reshaping operations.
        Grid/window reverse (Unblock/Ungrid) is performed on the final output for the same reason.
    Args:
        in_channels (int): Number of input channels.
        partition_function (Callable): Partition function to be utilized (grid or window partition).
        reverse_function (Callable): Reverse function to be utilized  (grid or window reverse).
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
    """

    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(GBTrans, self).__init__()
        # Save parameters
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        self.grid_window_size: Tuple[int, int] = grid_window_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            grid_window_size=grid_window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)].
        """
        # Save original shape
        B, C, H, W = input.shape
        # Perform partition
        input_partitioned = self.partition_function(input, self.grid_window_size)
        input_partitioned = input_partitioned.view(-1, self.grid_window_size[0] * self.grid_window_size[1], C)
        # Perform normalization, attention, and dropout
        output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
        # Perform normalization, MLP, and dropout
        output = output * self.drop_path(self.mlp(self.norm_2(output)))
        # Reverse partition
        output = self.reverse_function(output, (H, W), self.grid_window_size)
        return output


# -------------------------------------------------------------------------------------- #
class InvertibleGBTrans(nn.Module):
    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            mlp_ratio: float = 4.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            numTraceSamples: int = 1,
            numSeriesTerms: int = 5,
    ) -> None:
        super(InvertibleGBTrans, self).__init__()
        self.GBTrans_branch = GBTrans(in_channels, partition_function, reverse_function, num_heads, grid_window_size,
                                      attn_drop, drop, drop_path, mlp_ratio, act_layer, norm_layer)
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms

    def forward(self, x: torch.Tensor, ignore_logdet: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        Fx = self.GBTrans_branch(x)
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
                summand = self.GBTrans_branch(x)
                x = y - summand
            return x


# -------------------------------------------------------------------------------------- #
def window_partition(input: torch.Tensor, window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    """ Window partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)
    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, original_size: Tuple[int, int],
                   window_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width: 14x14
    H, W = original_size
    # Compute original batch size: 7
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def grid_partition(input: torch.Tensor, grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    """ Grid partition function.
    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        grid_size (Tuple[int, int], optional): Grid size to be applied. Default (7, 7)
    Returns:
        grid (torch.Tensor): Unfolded input tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    grid = input.view(B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1])
    # Permute and reshape [B * (H // grid_size[0]) * (W // grid_size[1]), grid_size[0], window_size[1], C]
    grid = grid.permute(0, 3, 5, 2, 4, 1).contiguous().view(-1, grid_size[0], grid_size[1], C)
    return grid


def grid_reverse(grid: torch.Tensor, original_size: Tuple[int, int],
                 grid_size: Tuple[int, int] = (7, 7)) -> torch.Tensor:
    """ Reverses the grid partition.
    Args:
        Grid (torch.Tensor): Grid tensor of the shape [B * grids, grid_size[0], grid_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        grid_size (Tuple[int, int], optional): Grid size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height, width, and channels, pay attention to grid, add the channels to computer. 3x14x14
    (H, W), C = original_size, grid.shape[-1]
    # Compute original batch size: 7
    B = int(grid.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # Fold grid tensor
    output = grid.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], C)
    output = output.permute(0, 5, 3, 1, 4, 2).contiguous().view(B, C, H, W)
    return output


###################################################################
if __name__ == '__main__':

    def test_partition_and_revers() -> None:  # the shape of windows and grid is same. 7x3x14x14
        input = torch.rand(7, 3, 14, 14)
        windows = window_partition(input=input)
        windows = window_reverse(windows=windows, window_size=(7, 7), original_size=input.shape[2:])
        print(torch.all(input == windows))
        grid = grid_partition(input=input)
        grid = grid_reverse(grid=grid, grid_size=(7, 7), original_size=input.shape[2:])
        print(torch.all(input == grid))

    def test_GBTrans() -> None:
        gbtrans_grid = GBTrans(in_channels=128, partition_function=grid_partition, reverse_function=grid_reverse)
        input = torch.rand(4, 128, 7, 7)
        output_grid = gbtrans_grid(input)
        print('the shape of GBTrans for grid partition:', output_grid.shape)
        gbtrans_win = GBTrans(in_channels=128, partition_function=window_partition, reverse_function=window_reverse)
        output_win = gbtrans_win(input)
        print('the shape of GBTrans for window partition:', output_win.shape)

    def test_InvGBTrans() -> None:
        inv_gbtrans_grid = InvertibleGBTrans(in_channels=128, partition_function=grid_partition,
                                             reverse_function=grid_reverse)
        input = torch.rand(4, 128, 7, 7).requires_grad_()
        output_inv_grid = inv_gbtrans_grid.forward(input)
        print('the shape of Invertible GBTrans for grid partition:', output_inv_grid[0].shape)
        inv_gbtrans_win = InvertibleGBTrans(in_channels=128, partition_function=window_partition,
                                             reverse_function=window_reverse)
        output_inv_win = inv_gbtrans_win.inverse(input)
        print('the shape of Invertible GBTrans for window partition:', output_inv_win.shape)


    # UnitTest
    test_GBTrans()
    test_InvGBTrans()
    test_partition_and_revers()
