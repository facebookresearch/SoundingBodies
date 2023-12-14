"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch.nn as nn
from .builder import NETS


@NETS.register_module()
class PoseConv2DNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=32,
                 out_channels=128,
                 num_keypoints=7,
                 kernel_size=8):
        super().__init__()
        print(f"create {self.__class__.__name__}")
        self.posenet = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(kernel_size, 1)),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=(kernel_size, 1)),
        )
        self.out = nn.Conv1d(base_channels*num_keypoints, out_channels, kernel_size=1)

    def forward(self, body_pose):
        body_pose = body_pose.permute(0, 3, 1, 2).contiguous()  # N, C, T, V
        body_pose = self.posenet(body_pose)  # N, C', T', V
        N, C, T, V = body_pose.shape
        body_pose = body_pose.permute(0, 3, 1, 2).reshape(N, V*C, T).contiguous()
        body_pose = self.out(body_pose)

        return body_pose
