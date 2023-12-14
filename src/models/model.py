"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os.path as osp
import torch as th
import torch.nn as nn
from .builder import MODELS, build_net
from .basicblock import TimeWarper


@MODELS.register_module()
class AudioPoseModel(nn.Module):
    def __init__(self, audio_net, pose_net, lr, pre_pose_len, use_cuda=True):
        super().__init__()
        self.geowarper = TimeWarper()
        self.audio_net = build_net(audio_net)
        self.pose_net = build_net(pose_net)
        self.lr = lr
        self.pre_pose_len = pre_pose_len
        if use_cuda:
            self.audio_net.cuda()
            self.pose_net.cuda()

    @property
    def weights(self):
        weights = [
            {'params': self.audio_net.parameters(), 'name': 'audio_net', 'lr': self.lr['audio']},
            {'params': self.pose_net.parameters(), 'name': 'pose_net', 'lr': self.lr['pose']}
        ]

        return weights

    def save(self, model_dir, suffix=''):
        th.save(
            {'audio_net': self.audio_net.state_dict(),
             'pose_net': self.pose_net.state_dict()},
            osp.join(model_dir, f'model-{suffix}.pth')
        )

    def load(self, model_dir, device='cuda', suffix='', load_audio_only=False, strict=True):
        print(f'Loading checkpoint model-{suffix} from {model_dir}.')
        state_dict = th.load(osp.join(model_dir, f'model-{suffix}.pth'), map_location=device)
        self.audio_net.load_state_dict(state_dict['audio_net'], strict=strict)
        if not load_audio_only:
            self.pose_net.load_state_dict(state_dict['pose_net'], strict=strict)

    def hmc_warping(self, hmc_audio, shift_dist):
        B, C, L = hmc_audio.shape
        if shift_dist.dim() == 4:
            N = shift_dist.shape[2]
            K = shift_dist.shape[-1]
            hmc_audio = hmc_audio.unsqueeze(1).repeat(1, N * K, 1, 1).reshape(B * N * K, C, L)
            shift_dist = shift_dist.permute(0, 2, 3, 1).reshape(B * N * K, -1)
            shift_dist = shift_dist.unsqueeze(1).repeat(1, C, 1)
            hmc_shifted = self.geowarper(hmc_audio, shift_dist)
            hmc_shifted = hmc_shifted.reshape(B * N, C * K, L).float()
        else:
            N = shift_dist.shape[-1]
            hmc_audio = hmc_audio.unsqueeze(1).repeat(1, N, 1, 1).reshape(B*N, C, L)
            shift_dist = shift_dist.permute(0, 2, 1).reshape(B*N, -1)
            hmc_shifted = self.geowarper(hmc_audio, shift_dist)
        return hmc_shifted

    def forward(self, hmc_audio, body_pose, mics, shift_dist):
        hmc_audio = self.hmc_warping(hmc_audio, shift_dist)
        head_pose = body_pose[:, self.pre_pose_len:, :5]
        pose_feat = self.pose_net(body_pose)
        pred_audio = self.audio_net(hmc_audio, pose_feat, head_pose, mics)

        return pred_audio
