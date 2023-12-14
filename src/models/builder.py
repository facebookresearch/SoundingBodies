"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from mmcv.utils import Registry, build_from_cfg

MODELS = Registry('models')
NETS = MODELS


def build_net(cfg):
    return NETS.build(cfg)


def build_model(cfg):
    return MODELS.build(cfg)
