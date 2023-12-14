"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as th
import torch.nn as nn
import math
from .utils import FourierTransform


class Loss(nn.Module):
    def __init__(self, mask_beginning=0, mask_end=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning
        self.mask_end = mask_end

    def forward(self, data, target, **kwargs):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data_len = data.shape[-1]
        data = data[..., self.mask_beginning : data_len - self.mask_end]
        target = target[..., self.mask_beginning : data_len - self.mask_end]
        return self._loss(data, target, **kwargs)

    def _loss(self, data, target):
        raise NotImplementedError


class L1Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean(th.abs(data - target))


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean((data - target).pow(2))


class NormWeightedL2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean((data - target).pow(2) * weight) * min(5., 1/weight.max())


class WeightedL1Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean(th.abs(data - target) * weight)


class WeightedL2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        assert len(target.shape) == 3
        weight = th.abs(target)
        return th.mean((data - target).pow(2) * weight)


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, mask_end=0, log=False):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning, mask_end)
        self.log = log
        self.eps = 1e-8
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        data = th.sqrt(th.sum(data**2, dim=-1) + self.eps)
        target = th.sqrt(th.sum(target**2, dim=-1) + self.eps)
        if self.log:
            data = th.log(data + 1.0)
            target = th.log(target + 1.0)
        return th.mean(th.abs(data - target))


class PhaseLoss(Loss):
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        fft_bins=2048,
        ignore_below=0.1,
        silent_threshold=0.001,
        weighted=False
    ):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning, mask_end)
        self.ignore_below = ignore_below
        self.silent_threshold = silent_threshold
        self.weighted = weighted
        self.eps = 1e-8
        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        fft_params = {'fft_bins': fft_bins}
        fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
        fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
        fft_params['sample_rate'] = sample_rate
        self.fft = FourierTransform(**fft_params)

    @staticmethod
    def _get_silence_mask(audio, threshold, window_size=65):
        pooling = nn.MaxPool1d(window_size, stride=1)
        pad_len = window_size//2
        mask = pooling(nn.functional.pad(audio.abs(), (pad_len, pad_len), mode='replicate')) < threshold

        return mask

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data.clone()
        target = target.clone()
        silence_mask = self._get_silence_mask(target, self.silent_threshold)
        data[silence_mask] = 0
        target[silence_mask] = 0

        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2)
        # ignore low energy components for numerical stability
        target_energy = th.sqrt(th.sum(target**2, dim=-1) + self.eps)
        pred_energy = th.sqrt(th.sum(data.detach()**2, dim=-1) + self.eps)
        target_mask = target_energy > self.ignore_below * th.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * th.mean(target_energy)
        indices = th.nonzero(target_mask * pred_mask).view(-1)
        if len(indices) == 0:
            return th.ones(1)[0].to(data.device) * th.pi
        data, target = th.index_select(data, 0, indices), th.index_select(target, 0, indices)
        target_energy = th.index_select(target_energy, 0, indices)
        # compute actual phase loss in angular space
        data_angles, target_angles = th.atan2(data[:, 0], data[:, 1]), th.atan2(target[:, 0], target[:, 1])
        loss = th.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - th.abs(loss - np.pi)
        if self.weighted:
            return th.mean(loss) * th.mean(target_energy)
        else:
            return th.mean(loss)


class MultiResolutionSTFTLoss(Loss):
    """Multi resolution STFT loss module."""
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        log=False,
        fft_bins_list=[256, 128, 64, 32],
        weighted=False
    ):
        super().__init__(mask_beginning, mask_end=mask_end)
        self.log = log
        self.eps = 1e-8
        self.weighted = weighted

        assert len(fft_bins_list) >= 3
        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        self.fft_list = []
        for fft_bins in fft_bins_list:
            curr_fft_params = {'fft_bins': fft_bins}
            curr_fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
            curr_fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
            curr_fft_params['sample_rate'] = sample_rate
            self.fft_list.append(FourierTransform(**curr_fft_params))

    def _loss(self, data, target):
        data = data.view(-1, data.shape[-1])  # (B, C, T) -> (B x C, T)
        target = target.view(-1, target.shape[-1])  # (B, C, T) -> (B x C, T)
        mag_loss = 0.0
        for fft in self.fft_list:
            data_stft = fft.stft(data)
            target_stft = fft.stft(target)
            data_stft = th.sqrt(th.sum(data_stft**2, dim=-1) + self.eps)
            target_stft = th.sqrt(th.sum(target_stft**2, dim=-1) + self.eps)
            if self.log:
                data_stft = th.log(data_stft + 1.0)
                target_stft = th.log(target_stft + 1.0)
            if self.weighted:
                mag_loss += th.mean(th.abs(data_stft - target_stft) * target_stft)
            else:
                mag_loss += th.mean(th.abs(data_stft - target_stft))
        
        mag_loss /= len(self.fft_list)

        return mag_loss


class STFTLoss(Loss):
    def __init__(
        self,
        sample_rate,
        mask_beginning=0,
        mask_end=0,
        log=False,
        fft_bins=2048,
        weighted=False
    ):
        super().__init__(mask_beginning, mask_end=mask_end)
        self.log = log
        self.eps = 1e-8
        self.weighted = weighted

        fft_bins_ref = 2048
        win_length_ms_ref = 40
        frame_rate_hz_ref = 100
        fft_params = {'fft_bins': fft_bins}
        fft_params['win_length_ms'] = win_length_ms_ref * fft_bins / fft_bins_ref
        fft_params['frame_rate_hz'] = frame_rate_hz_ref * fft_bins_ref / fft_bins
        fft_params['sample_rate'] = sample_rate
        self.fft = FourierTransform(**fft_params)

    def _loss(self, data, target):
        data = data.view(-1, data.shape[-1])  # (B, C, T) -> (B x C, T)
        target = target.view(-1, target.shape[-1])  # (B, C, T) -> (B x C, T)
        data_stft = self.fft.stft(data)
        target_stft = self.fft.stft(target)
        data_stft = th.sqrt(th.sum(data_stft**2, dim=-1) + self.eps)
        target_stft = th.sqrt(th.sum(target_stft**2, dim=-1) + self.eps)
        if self.log:
            data_stft = th.log(data_stft + 1.0)
            target_stft = th.log(target_stft + 1.0)
        if self.weighted:
            mag_loss = th.mean(th.abs(data_stft - target_stft) * target_stft)
        else:
            mag_loss = th.mean(th.abs(data_stft - target_stft))

        return mag_loss


class ShiftedL2Loss(Loss):
    def __init__(self, sample_rate, alpha=100, max_dist=1.5, mask_beginning=0, mask_end=0, device='cuda'):
        super().__init__(mask_beginning, mask_end)
        dist_samples = int(sample_rate * max_dist / 340)                                # max dist in samples
        n = math.floor(math.log(dist_samples, 2))                                       # closest power of two
        self.t = 2 ** n                                                                 # sequence length (short)
        self.S = self.t * 2 + 1                                                         # shifts length
        self.T = self.t + self.S - 1                                                    # sequence length (long)
        self.dT = int(self.t)                                                           # step size
        self.w = alpha * (1 - (th.blackman_window(self.S, periodic=False).to(device)))
        self.device = device
        self.N = 0
        self.padding = 0

    def _compute(self, x, y):
        y = th.cat([y[:, int(i * self.dT): int(i * self.dT + self.T)]
                    for i in range(self.N)], dim=0)                                     # shape: [M * N x T]
        x = th.cat([x[:, int(i * self.dT): int(i * self.dT + self.t)]
                    for i in range(self.N)], dim=0)                                     # shape: [M * N x t]
        loss = th.stack([th.mean((x - y[:, k:k + x.shape[-1]]) ** 2, dim=1) for k in range(self.S)], dim=0)

        shift_win = self.w.unsqueeze(1).expand(-1, loss.shape[1])
        loss = (loss + 1) * (shift_win + 1) - 1                                         # shape: [S x M * N]
        loss = th.amin(loss, dim=0, keepdim=True)                                       # shape: [1 x M * N]

        loss = th.mean(loss, dim=1)                                                     # shape: [1]
        return loss

    def _loss(self, data, ref):
        """
        :param data: th.Tensor of shape ... x L (where L is the time dimension) representing the predicted signal
        :param ref: th.Tensor of shape ... x L (where L is the time dimension) representing the reference/gt signal
        """
        # e.g. data shape: [B, C, L] => M = B * C
        data = data.view(-1, data.shape[-1])                                            # shape: [M x L]
        ref = ref.view(-1, ref.shape[-1])                                               # shape: [M x L]
        self.N = int(np.floor((data.shape[-1] - self.t) / self.dT)) + 1                 # number of sequences
        data = data[:, : (self.N - 1) * self.dT + self.t]
        ref = ref[:, : (self.N - 1) * self.dT + self.t]

        # Normalization
        d = data[:, self.t:-self.t]
        r = ref[:, self.t:-self.t]
        data_mean = th.mean(d, dim=1, keepdim=True).expand(-1, d.shape[1])              # shape: [M x L]
        ref_mean = th.mean(r, dim=1, keepdim=True).expand(-1, r.shape[1])               # shape: [M x L]
        d = d - data_mean
        r = r - ref_mean

        mag_data = th.sqrt(th.mean(d ** 2, dim=1, keepdim=True))                        # shape: [M x 1]
        mag_ref = th.sqrt(th.mean(r ** 2, dim=1, keepdim=True))                         # shape: [M x 1]
        mag_data = th.amin(th.cat([mag_data, mag_ref], dim=1), dim=1, keepdim=True)
        mag_norm = th.sqrt(mag_data * mag_ref).expand(-1, data.shape[-1]) + 0.001       # shape: [M x L]
        data = data / mag_norm                                                          # shape: [M x L]
        ref = ref / mag_norm                                                            # shape: [M x L]

        # Compute
        self.padding = th.zeros((data.shape[0], self.t)).to(self.device)
        ref = th.cat([self.padding, ref, self.padding], dim=1)                          # shape: [M x L + 2 * t]
        loss = self._compute(data, ref)

        return loss


class SDRloss(Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _loss(self, data, ref):
        return th.mean(th.log10(th.mean(ref**2, dim=-1) / th.mean((ref-data)**2, dim=-1))) * 10
