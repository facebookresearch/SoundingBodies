"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn as nn
from .hyperconv import HyperConvBlock, ConvBlock, ConvMulBlock


class WaveNet(nn.Module):
    def __init__(self,
                 channels=512,
                 pose_dome_dim=256,
                 blocks=3,
                 layers_per_block=10,
                 conv_kernel=2,
                 causal=True,
                 convblock_name='ConvBlock'):
        super().__init__()
        self.layers = []
        self.rectv_field = 1
        self.causal = causal
        if convblock_name == 'ConvBlock':
            convblock = ConvBlock
        elif convblock_name == 'ConvMulBlock':
            convblock = ConvMulBlock
        else:
            convblock = HyperConvBlock
        for b in range(blocks):
            for l in range(layers_per_block):
                self.layers += [convblock(channels, pose_dome_dim, kernel_size=conv_kernel, dilation=2**l, causal=causal)]
                self.rectv_field += self.layers[-1].receptive_field() - 1
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, pose_dome):
        skips = []
        for layer in self.layers:
            x, skip = layer(x, pose_dome)
            skips += [skip]
        return x, skips

    def receptive_field(self):
        return self.rectv_field


class WaveoutBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size=1)
        self.second = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x):
        x = th.relu(self.first(x))
        return self.second(x).clamp(min=-1, max=1)


class TimeWarperFunction(th.autograd.Function):

    @staticmethod
    def forward(ctx, input, warpfield):
        '''
        :param ctx: autograd context
        :param input: input signal (B x 2 x T)
        :param warpfield: the corresponding warpfield (B x 2 x T)
        :return: the warped signal (B x 2 x T)
        '''
        ctx.save_for_backward(input, warpfield)
        # compute index list to lookup warped input values
        idx_left = warpfield.floor().type(th.long)
        idx_right = th.clamp(warpfield.ceil().type(th.long), max=input.shape[-1]-1)
        # compute weight for linear interpolation
        alpha = warpfield - warpfield.floor()
        # linear interpolation
        output = (1 - alpha) * th.gather(input, 2, idx_left) + alpha * th.gather(input, 2, idx_right)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, warpfield = ctx.saved_tensors
        # compute index list to lookup warped input values
        idx_left = warpfield.floor().type(th.long)
        idx_right = th.clamp(warpfield.ceil().type(th.long), max=input.shape[-1]-1)
        # warpfield gradient
        grad_warpfield = th.gather(input, 2, idx_right) - th.gather(input, 2, idx_left)
        grad_warpfield = grad_output * grad_warpfield
        # input gradient
        grad_input = th.zeros(input.shape, device=input.device)
        alpha = warpfield - warpfield.floor()
        grad_input = grad_input.scatter_add(2, idx_left, grad_output * (1 - alpha)) + \
                     grad_input.scatter_add(2, idx_right, grad_output * alpha)
        return grad_input, grad_warpfield


class TimeWarper(nn.Module):

    def __init__(self):
        super().__init__()
        self.warper = TimeWarperFunction().apply

    def _to_absolute_positions(self, warpfield, seq_length):
        # translate warpfield from relative warp indices to absolute indices ([1...T] + warpfield)
        temp_range = th.arange(seq_length, dtype=th.float)
        temp_range = temp_range.cuda() if warpfield.is_cuda else temp_range
        return th.clamp(warpfield + temp_range[None, None, :], min=0, max=seq_length-1)

    def forward(self, input, warpfield):
        '''
        :param input: audio signal to be warped (B x 2 x T)
        :param warpfield: the corresponding warpfield (B x 2 x T)
        :return: the warped signal (B x 2 x T)
        '''
        B, C, L = input.shape
        assert input.shape[-1] % warpfield.shape[-1] == 0
        times = input.shape[-1] // warpfield.shape[-1]
        warpfield = warpfield.unsqueeze(-1).repeat(1,1,1,times).reshape(B, C, L)
        warpfield = self._to_absolute_positions(warpfield, input.shape[-1])
        warped = self.warper(input, warpfield)
        return warped
