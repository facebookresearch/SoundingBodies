"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import os.path as osp
import argparse
import git
import shutil
import warnings
warnings.filterwarnings("ignore", message="On January 1, 2023, MMCV will release v2.0.0")
from mmcv import Config
from src.trainer import TrainerDp, TrainerDdp
from src.datasets import build_dataset
from src.models import build_model
from src.utils import set_random_seed

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config",
                    type=str,
                    default="./config/config.py",
                    help="path to the config file")
parser.add_argument("-b", "--branch",
                    type=str,
                    default="",
                    help="git switch to which branch")
parser.add_argument("--seed",
                    type=int,
                    default=1234,
                    help="set the random seed")
args = parser.parse_args()
set_random_seed(seed=args.seed, deterministic=False)

configs = Config.fromfile(args.config)
os.makedirs(configs.output_dir, exist_ok=True)
shutil.copy(args.config, configs.output_dir)
repo = git.Repo(search_parent_directories=True)
if len(args.branch) > 0:
    repo.git.checkout(args.branch)
    print("Now git switch to branch", args.branch)
with open(osp.join(configs.output_dir, 'train_info.txt'), 'w') as f:
    f.write("code version:\n")
    f.write(repo.head.object.hexsha)
    f.write("\n")

dataset_train = build_dataset(configs.dataset_train)
print(f"train on {len(dataset_train.chunks)} chunks")
dataset_val = build_dataset(configs.dataset_val)
print(f"validate on {len(dataset_val.chunks)} chunks")
dataset = {'train': dataset_train, 'val': dataset_val, 'test': None}

model = build_model(configs.model)

trainer = TrainerDp(configs.training, model, dataset, seed=args.seed)  # python train.py --config config_file.py
# torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py --config config_file.py
# trainer = TrainerDdp(configs.training, model, dataset, seed=args.seed)

trainer.train()
print('Finished!')
