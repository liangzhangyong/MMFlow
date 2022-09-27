# ----------------------------------------------------- #
#    Written by zyliang (zyliang18@mails.jlu.edu.cn)    #
# ----------------------------------------------------- #

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from matrix_utils import power_series_matrix_logarithm_trace
from spectral_norm_fc import spectral_norm_fc

from typing import Type, Callable, Tuple, Optional, Set, List, Union, Any
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath

__all__ = ['SelfAttention',    'RelativeSelfAttention',
           'InvSelfAttention', 'InvRelativeSelfAttention']


# -------------------------------------------------------------------------------------- #
# baseline self-attention
class SelfAttention(nn.Module):
    """
    scale dot-product attention, inverse
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 8,
            drop: float = 0.,
            LipNorm=True,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(SelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        self.LipNorm: bool = LipNorm
        # Init layers
        self.norm = nn.LayerNorm(in_channels)
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.qkv_mapping = spectral_norm_fc(self.qkv_mapping, coeff=.9, n_power_iterations=5)  # 保证可逆，需要规范化
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj = spectral_norm_fc(self.proj, coeff=.9, n_power_iterations=5)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        self.nonlin = nn.GELU()
        if LipNorm:
            self.gamma = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
            self.gamma = spectral_norm_fc(self.gamma, coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get shape of input
        B_, N, C = input.shape
        # normlization of input
        input = self.norm(input)
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # Scale query
        query = query * self.scale
        # Compute attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...kj->...ik', a, b)
        # attn = self.softmax((query @ key.transpose(-2, -1)))
        attn = self.softmax(bmmp(query, key))
        # Lipschitz normlization
        if self.LipNorm:
            print('normlization')
        # Map values with attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...jk->...ik', a, b)
        # output = (attn @ value).transpose(1, 2).reshape(B_, N, -1)
        output = bmmp(attn, value).transpose(1, 2).reshape(B_, N, -1)
        # Lipschitz Normlization
        if self.LipNorm:
            output = self.gamma(output)
        else:
            output = self.nonlin(self.gamma) * output
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


# --------------------------------
class InvSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_head: int = 8,
            drop: float = 0.,
            numTraceSamples: int = 1,
            numSeriesTerms: int = 5,
            LipNorm: bool = True,
    ) -> None:
        super(InvSelfAttention, self).__init__()
        self.attn_branch = SelfAttention(in_channels, num_head, drop, LipNorm)
        self.numTraceSamples: int = numTraceSamples
        self.numSeriesTerms: int = numSeriesTerms

    def forward(self, x: torch.Tensor, ignore_logdet: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        Fx = self.attn_branch(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        # add residual to output
        x = x + Fx
        return x, trace

    def inverse(self, y: torch.Tensor, maxIter: int = 100) -> torch.Tensor:
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)  need to modify
            x = y
            for iter_index in range(maxIter):
                summand = self.attn_branch(x)
                x = y - summand
            return x


# -------------------------------------------------------------------------------------- #
class NystromSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 8,
            num_landmarks: int = 7,
            drop: float = 0.,
            LipNorm=True,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(NystromSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.num_landmarks = num_landmarks
        self.scale: float = num_heads ** -0.5
        self.LipNorm: bool = LipNorm
        # Init layers
        self.norm = nn.LayerNorm(in_channels)
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)  # 使用conv2D也可以，kernel=1
        self.qkv_mapping = spectral_norm_fc(self.qkv_mapping, coeff=.9, n_power_iterations=5)  # 保证可逆，需要规范化
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj = spectral_norm_fc(self.proj, coeff=.9, n_power_iterations=5)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        self.nonlin = nn.GELU()
        if LipNorm:
            self.gamma = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
            self.gamma = spectral_norm_fc(self.gamma, coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get shape of input
        B_, N, C = input.shape
        # normlization of input
        input = self.norm(input)
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # landmark mapping
        query_landmarks = query.reshape(B_, self.num_heads, self.num_landmarks, N // self.num_landmarks, C//self.num_heads).mean(dim=-2)
        key_landmarks = key.reshape(B_, self.num_heads, self.num_landmarks, N // self.num_landmarks, C//self.num_heads).mean(dim=-2)
        # Scale query
        query_landmarks = query_landmarks * self.scale
        key_landmarks = key_landmarks * self.scale

        # Compute attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...kj->...ik', a, b)
        attn1 = self.softmax(bmmp(query, key_landmarks))
        attn2 = self.softmax(bmmp(query_landmarks, key_landmarks))
        attn3 = self.softmax(bmmp(query_landmarks, key))
        attn2_inv = self.interative_inv(attn2)

        # Lipschitz normlization
        if self.LipNorm:
            print('normlization')
        # Map values with attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...jk->...ik', a, b)
        output = bmmp(bmmp(attn1, attn2_inv), bmmp(attn3, value)).transpose(1, 2).reshape(B_, N, -1)

        # Lipschitz Normlization
        if self.LipNorm:
            output = self.gamma(output)
        else:
            output = self.nonlin(self.gamma) * output
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

    def interative_inv(self, mat, n_iter=6):
        def matmul(A, B):
            return torch.einsum('...ij,...jk->...ik', A, B)
        Id = torch.eye(mat.size(-1), device=mat.device)
        K = mat
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        KA = K.abs()
        V0 = KA.sum(dim=-2, keepdim=True).amax(dim=-1, keepdim=True)
        VI = KA.sum(dim=-1, keepdim=True).amax(dim=-2, keepdim=True)
        V = K.transpose(-2, -1).div(V0 * VI)
        for _ in range(n_iter):
            KV = matmul(K, V)
            V = matmul(0.25 * V, 13 * Id - matmul(KV, 15 * Id - matmul(KV, 7 * Id - KV)))
        return V


# --------------------------------
class InvNystromSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_head: int = 8,
            num_landmarks: int = 7,
            drop: float = 0.,
            numTraceSamples: int = 1,
            numSeriesTerms: int = 5,
            LipNorm: bool = True,
    ) -> None:
        super(InvNystromSelfAttention, self).__init__()
        self.attn_branch = NystromSelfAttention(in_channels, num_head, num_landmarks, drop, LipNorm)
        self.numTraceSamples: int = numTraceSamples
        self.numSeriesTerms: int = numSeriesTerms

    def forward(self, x: torch.Tensor, ignore_logdet: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        Fx = self.attn_branch(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        # add residual to output
        x = x + Fx
        return x, trace

    def inverse(self, y: torch.Tensor, maxIter: int = 100) -> torch.Tensor:
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)  need to modify
            x = y
            for iter_index in range(maxIter):
                summand = self.attn_branch(x)
                x = y - summand
            return x


# -------------------------------------------------------------------------------------- #
class ParamixerSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 8,
            drop: float = 0.,
            LipNorm=True,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(ParamixerSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        self.LipNorm: bool = LipNorm
        # Init layers
        self.norm = nn.LayerNorm(in_channels)
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.qkv_mapping = spectral_norm_fc(self.qkv_mapping, coeff=.9, n_power_iterations=5)  # 保证可逆，需要规范化
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj = spectral_norm_fc(self.proj, coeff=.9, n_power_iterations=5)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        self.nonlin = nn.GELU()
        if LipNorm:
            self.gamma = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
            self.gamma = spectral_norm_fc(self.gamma, coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get shape of input
        B_, N, C = input.shape
        # normlization of input
        input = self.norm(input)
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # Scale query
        query = query * self.scale
        # Compute attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...kj->...ik', a, b)
        attn = self.softmax(bmmp(query, key))
        # Lipschitz normlization
        if self.LipNorm:
            print('normlization')
        # Map values with attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...jk->...ik', a, b)
        output = bmmp(attn, value).transpose(1, 2).reshape(B_, N, -1)
        # Lipschitz Normlization
        if self.LipNorm:
            output = self.gamma(output)
        else:
            output = self.nonlin(self.gamma) * output
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


# -------------------------------------------------------------------------------------- #
class RelativeSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            LipNorm: bool = True,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size: Tuple[int, int] = grid_window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = grid_window_size[0] * grid_window_size[1]
        self.LipNorm: bool = LipNorm
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.qkv_mapping = spectral_norm_fc(self.qkv_mapping, coeff=.9, n_power_iterations=5)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj = spectral_norm_fc(self.proj, coeff=.9, n_power_iterations=5)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_window_size[0] - 1) * (2 * grid_window_size[1] - 1), num_heads))
        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index",
                             get_relative_position_index(grid_window_size[0], grid_window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        # Lipschitz Normalization
        if LipNorm:
            self.gamma = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
            self.gamma = spectral_norm_fc(self.gamma, coeff=.9, n_power_iterations=5)
        else:
            self.gamma = Parameter(torch.zeros(1))
            self.nonlin_2 = nn.Tanh()

    def _get_relative_positional_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        # Scale query
        query = query * self.scale
        # Compute attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...kj->...ik', a, b)
        # attn = self.softmax(query @ key.transpose(-2, -1) + self._get_relative_positional_bias())
        attn = self.softmax(bmmp(query, key) + self._get_relative_positional_bias())
        # Map values with attention maps
        bmmp = lambda a, b: torch.einsum('...ij,...jk->...ik', a, b)
        # output = (attn @ value).transpose(1, 2).reshape(B_, N, -1)
        output = bmmp(attn, value).transpose(1, 2).reshape(B_, N, -1)
        # Lipschitz Norlization
        if self.LipNorm:
            output = self.gamma(output)
        else:
            output = self.nonlin(self.gamma) * output
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


# ------------------------------
class InvRelativeSelfAttention(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            grid_window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.,
            numTraceSamples: int = 1,
            numSeriesTerms: int = 5,
            LipNorm: bool = True,
    ) -> None:
        super(InvRelativeSelfAttention, self).__init__()
        self.attn_branch = RelativeSelfAttention(in_channels, num_heads, grid_window_size, attn_drop, drop, LipNorm)
        self.numTraceSamples: int = numTraceSamples
        self.numSeriesTerms: int = numSeriesTerms

    def forward(self, x: torch.Tensor, ignore_logdet: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        Fx = self.attn_branch(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
        # add residual to output
        x = x + Fx
        return x, trace

    def inverse(self, y: torch.Tensor, maxIter: int = 100) -> torch.Tensor:
        with torch.no_grad():
            # inversion of ResNet-block (fixed-point iteration)  need to modify
            x = y
            for iter_index in range(maxIter):
                summand = self.attn_branch(x)
                x = y - summand
            return x


# -------------------------------------------------------------------------------------- #
def get_relative_position_index(win_h: int, win_w: int) -> torch.Tensor:
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


###################################################################
if __name__ == '__main__':
    def test_self_attention() -> torch.Tensor:
        self_attention = SelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        output = self_attention.forward(input)
        print('the shape of self-attention:', output.shape)
        return output


    def test_inv_self_attention() -> tuple[tuple[Tensor, Tensor], Tensor]:
        from torch.autograd import Variable
        inv_self_attention = InvSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        input = Variable(input, requires_grad=True)
        output = inv_self_attention.forward(input)
        print('the shape of inv-self-attention forward:', output[0].shape)
        inv_output = inv_self_attention.inverse(input)
        print('the shape of inv-self-attention inverse:', inv_output.shape)
        return output, inv_output


    def test_relative_self_attention() -> torch.Tensor:
        relative_self_attention = RelativeSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        output = relative_self_attention.forward(input)
        print('the shape of relative self-attention:', output.shape)
        return output


    def test_inv_relative_self_attention() -> tuple[tuple[Tensor, Tensor], Tensor]:
        from torch.autograd import Variable
        inv_relative_self_attention = InvRelativeSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        input = Variable(input, requires_grad=True)
        output = inv_relative_self_attention.forward(input)
        print('the shape of inv-relative-self-attention forward:', output[0].shape)
        inv_output = inv_relative_self_attention.inverse(input)
        print('the shape of inv-relative-self-attention inverse:', inv_output.shape)
        return output, inv_output


    def test_nystrom_self_attention() -> torch.Tensor:
        nystrom_self_attention = NystromSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        output = nystrom_self_attention.forward(input)
        print('the shape of nystrom self-attention:', output.shape)
        return output


    def test_inv_nystrom_self_attention() -> tuple[tuple[Tensor, Tensor], Tensor]:
        from torch.autograd import Variable
        inv_nystrom_self_attention = InvNystromSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        input = Variable(input, requires_grad=True)
        output = inv_nystrom_self_attention.forward(input)
        print('the shape of inv-nystrom-self-attention forward:', output[0].shape)
        inv_output = inv_nystrom_self_attention.inverse(input)
        print('the shape of inv-nystrom-self-attention inverse:', inv_output.shape)
        return output, inv_output


    def test_paramixer_self_attention() -> torch.Tensor:
        paramixer_self_attention = ParamixerSelfAttention(in_channels=128)
        input = torch.rand(4, 7 * 7, 128)
        output = paramixer_self_attention.forward(input)
        print('the shape of paramixer self-attention:', output.shape)
        return output


    # UnitTest
    _self_attention = test_self_attention()
    _inv_self_attention = test_inv_self_attention()
    _re_self_attention = test_relative_self_attention()
    _inv_re_self_attention = test_inv_relative_self_attention()
    _nystrom_self_attention = test_nystrom_self_attention()
    _inv_nyt_self_attention = test_inv_nystrom_self_attention()

    print('the error of sefl-attention and relative self-attention:',
          _self_attention - _re_self_attention)
    print('the error of inverse self-attention and inverse relative self-attention in forward:',
          _inv_self_attention[0][0] - _inv_re_self_attention[0][0])
    print('the error of inverse self-attention and inverse relative self-attention in inverse:',
          _inv_self_attention[0][1] - _inv_re_self_attention[0][1])
    print('the error of sefl-attention and nystrom self-attention:',
          _self_attention - _nystrom_self_attention)
    print('the error of inverse self-attention and inverse nystrom self-attention in forward:',
          _inv_self_attention[0][0] - _inv_nyt_self_attention[0][0])
    print('the error of inverse self-attention and inverse relative self-attention in inverse:',
          _inv_self_attention[0][1] - _inv_nyt_self_attention[0][1])
