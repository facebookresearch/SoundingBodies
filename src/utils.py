"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import os.path as osp
import numpy as np
import torch as th
import torchaudio as ta
import random
import mmcv
from scipy import signal
import torch.distributed as dist


def set_random_seed(seed, deterministic=False, use_rank_shift=False):
    """
    Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = mmcv.runner.get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        th.backends.cudnn.benchmark = False
        th.use_deterministic_algorithms(True)
    else:
        th.backends.cudnn.benchmark = True


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
    th.cuda.set_device(local_rank)
    dist.barrier()


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


class Adam(th.optim.Adam):

    def __init__(self, weights, init_lr):
        super().__init__(weights, lr=init_lr)

    def update_lr(self, net, loss):
        pass


class AdamW(th.optim.AdamW):

    def __init__(self, weights, init_lr):
        super().__init__(weights, lr=init_lr)

    def update_lr(self, net, loss):
        pass


class NewbobAdam(th.optim.Adam):

    def __init__(self,
                 weights,
                 init_lr,
                 output_dir,
                 decay=0.5,
                 max_decay=0.01
                 ):
        '''
        Newbob learning rate scheduler
        :param weights: weights to optimize
        :param output_dir: (str) directory to save/restore models to/from
        :param init_lr: (float) initial learning rate
        :param decay: (float) value to decrease learning rate by when loss doesn't improve further
        :param max_decay: (float) maximum decay of learning rate
        '''
        super().__init__(weights, lr=init_lr)
        self.last_epoch_loss = np.inf
        self.total_decay = 1
        self.decay = decay
        self.max_decay = max_decay
        self.output_dir = output_dir

    def update_lr(self, net, loss):
        '''
        update the learning rate based on the current loss value and historic loss values
        :param loss: the loss after the current iteration
        '''
        if loss > self.last_epoch_loss and self.decay < 1.0 and self.total_decay > self.max_decay:
            self.total_decay = self.total_decay * self.decay
            print(f"NewbobAdam: Decay learning rate (loss degraded from {self.last_epoch_loss} to {loss})."
                  f"Total decay: {self.total_decay}")
            # restore previous network state
            net.load(self.output_dir, suffix="newbob")
            # decrease learning rate
            for param_group in self.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay
        else:
            self.last_epoch_loss = loss
        # save last snapshot to restore it in case of lr decrease
        if self.decay < 1.0 and self.total_decay > self.max_decay:
            net.save(self.output_dir, suffix="newbob")


class FourierTransform:
    def __init__(self,
                 fft_bins=2048,
                 win_length_ms=40,
                 frame_rate_hz=100,
                 causal=False,
                 preemphasis=0.0,
                 sample_rate=48000,
                 normalized=False):
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.preemphasis = preemphasis
        self.fft_bins = fft_bins
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate / frame_rate_hz)
        self.causal = causal
        self.normalized = normalized
        if self.win_length > self.fft_bins:
            print('FourierTransform Warning: fft_bins should be larger than win_length')

    def _convert_format(self, data, expected_dims):
        if not type(data) == th.Tensor:
            data = th.Tensor(data)
        if len(data.shape) < expected_dims:
            data = data.unsqueeze(0)
        if not len(data.shape) == expected_dims:
            raise Exception(f"FourierTransform: data needs to be a Tensor with {expected_dims} dimensions but got shape {data.shape}")
        return data

    def _preemphasis(self, audio):
        if self.preemphasis > 0:
            return th.cat((audio[:, 0:1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1)
        return audio

    def _revert_preemphasis(self, audio):
        if self.preemphasis > 0:
            for i in range(1, audio.shape[1]):
                audio[:, i] = audio[:, i] + self.preemphasis * audio[:, i-1]
        return audio

    def _magphase(self, complex_stft):
        mag, phase = ta.functional.magphase(complex_stft, 1.0)
        return mag, phase

    def stft(self, audio):
        '''
        wrapper around th.stft
        audio: wave signal as th.Tensor
        '''
        hann = th.hann_window(self.win_length)
        hann = hann.cuda() if audio.is_cuda else hann
        # spec = th.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
        #                window=hann, center=not self.causal, normalized=self.normalized, return_complex=False)
        spec = th.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
                       window=hann, center=not self.causal, normalized=self.normalized, return_complex=True)
        spec = th.view_as_real(spec)
        return spec.contiguous()

    def complex_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: th.Tensor of size channels x frequencies x time_steps (channels x y_axis x x_axis)
        '''
        self._convert_format(audio, expected_dims=2)
        audio = self._preemphasis(audio)
        return self.stft(audio)

    def magnitude_phase(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: tuple containing two th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        stft = self.complex_spectrogram(audio)
        return self._magphase(stft)

    def mag_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[0]

    def power_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: power spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return th.pow(self.mag_spectrogram(audio), 2.0)

    def phase_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: phase spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[1]

    def mel_spectrogram(self, audio, n_mels):
        '''
        audio: wave signal as th.Tensor
        n_mels: number of bins used for mel scale warping
        return: mel spectrogram as th.Tensor of size channels x n_mels x time_steps for magnitude and phase spectrum
        '''
        spec = self.power_spectrogram(audio)
        mel_warping = ta.transforms.MelScale(n_mels, self.sample_rate)
        return mel_warping(spec)

    def complex_spec2wav(self, complex_spec, length):
        '''
        inverse stft
        complex_spec: complex spectrum as th.Tensor of size channels x frequencies x time_steps x 2 (real part/imaginary part)
        length: length of the audio to be reconstructed (in frames)
        '''
        complex_spec = self._convert_format(complex_spec, expected_dims=4)
        hann = th.hann_window(self.win_length)
        hann = hann.cuda() if complex_spec.is_cuda else hann
        wav = ta.functional.istft(complex_spec, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length, window=hann, length=length, center=not self.causal)
        wav = self._revert_preemphasis(wav)
        return wav

    def magphase2wav(self, mag_spec, phase_spec, length):
        '''
        reconstruction of wav signal from magnitude and phase spectrum
        mag_spec: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps
        phase_spec: phase spectrum as th.Tensor of size channels x frequencies x time_steps
        length: length of the audio to be reconstructed (in frames)
        '''
        mag_spec = self._convert_format(mag_spec, expected_dims=3)
        phase_spec = self._convert_format(phase_spec, expected_dims=3)
        complex_spec = th.stack([mag_spec * th.cos(phase_spec), mag_spec * th.sin(phase_spec)], dim=-1)
        return self.complex_spec2wav(complex_spec, length)
