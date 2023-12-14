"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch as th
import torch.nn as nn
from .builder import NETS
from .basicblock import WaveNet, WaveoutBlock


@NETS.register_module()
class AudioHeadPoseNet(nn.Module):
    def __init__(self,
                 input_channels=7,
                 pose_embed_dim=32,
                 head_embed_dim=16,
                 mic_embed_dim=16,
                 wavenet_blocks=3,
                 layers_per_block=10,
                 wavenet_channels=64,
                 conv_kernel=2,
                 causal=True,
                 convblock_name='ConvBlock',
                 ):
        super().__init__()
        if head_embed_dim > 0:
            self.head_embed = nn.Linear(5 * 3, head_embed_dim)
        else:
            self.head_embed = None
        self.mic_embed = nn.Linear(3, mic_embed_dim)
        pose_dome_dim = pose_embed_dim + head_embed_dim + mic_embed_dim
        self.input = nn.Conv1d(input_channels, wavenet_channels, kernel_size=1)
        self.wavenet = WaveNet(
            channels=wavenet_channels,
            pose_dome_dim=pose_dome_dim,
            blocks=wavenet_blocks,
            layers_per_block=layers_per_block,
            conv_kernel=conv_kernel,
            causal=causal,
            convblock_name=convblock_name,
        )
        outblock = WaveoutBlock
        self.output_net = nn.ModuleList([outblock(wavenet_channels)
                                         for _ in range(wavenet_blocks * layers_per_block)])
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"create {self.__class__.__name__}")
        print(f"receptive field: {self.receptive_field()}")
        print(f"number of trainable parameters: {self.num_params}")

    def forward(self, hmc_audio, pose_feat, head_enc, mic_enc):
        x = self.input(hmc_audio)
        B_N, C, T = x.shape
        B, K, _, _ = head_enc.shape
        N = mic_enc.shape[1]
        assert T % K == 0
        assert B == B_N // N

        if pose_feat is not None:
            p_embed = pose_feat.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, K).contiguous()

        if self.head_embed is not None:
            h_embed = self.head_embed(head_enc.reshape(B * K, -1))                             # shape: [B*K, C_h]
            h_embed = h_embed.view(B, K, -1).transpose(1, 2)                                   # shape: [B, C_h, K]
            h_embed = h_embed.unsqueeze(1).repeat(1, N, 1, 1).view(B * N, -1, K).contiguous()  # shape: [BxN, C_h, K]

        m_embed = self.mic_embed(mic_enc.reshape(B * N, -1))                                   # shape: [B*N, C_m]
        m_embed = m_embed.unsqueeze(-1).expand(-1, -1, K)                                      # shape: [B*N, C_m, K]

        if self.head_embed is not None and pose_feat is not None:
            all_embed = th.cat([m_embed, h_embed, p_embed], dim=1)
        elif self.head_embed is not None and pose_feat is None:
            all_embed = th.cat([m_embed, h_embed], dim=1)
        elif self.head_embed is None and pose_feat is not None:
            all_embed = th.cat([m_embed, p_embed], dim=1)
        else:
            all_embed = m_embed

        _, skips = self.wavenet(x, all_embed)
        k = len(skips)
        y = th.mean(th.stack(skips[:k], dim=0), dim=0)
        y = self.output_net[k - 1](y)
        y = y.squeeze(1).reshape(B, N, -1)
        x = y
        return {"output": x}

    def receptive_field(self):
        return self.wavenet.receptive_field()
