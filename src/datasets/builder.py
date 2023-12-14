"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from mmcv.utils import Registry, build_from_cfg

DATASETS = Registry('dataset')


def build_dataset(cfg):
    return DATASETS.build(cfg)
