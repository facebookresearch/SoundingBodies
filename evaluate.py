"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore", message="On January 1, 2023, MMCV will release v2.0.0")
from mmcv import Config
import torchaudio as ta
from src.datasets import build_dataset
from src.trainer import TrainerDp
from src.utils import set_random_seed
from src.models import build_model
from src.losses import L2Loss, STFTLoss, PhaseLoss, ShiftedL2Loss
import datetime
import copy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        type=str,
                        default="./config/config.py",
                        help="path to the config file")
    parser.add_argument("-e", "--test_epoch",
                        type=str,
                        default="epoch-100",
                        help="which checkpoint to load")
    parser.add_argument("-s", "--save",
                        action="store_true",
                        help="whether save the synthesized audio")
    parser.add_argument("-n", "--out_name",
                        type=str,
                        # default='metrics',
                        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help="the output filename of metrics")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="set the random seed")
    args = parser.parse_args()
    set_random_seed(seed=args.seed, deterministic=False)

    configs = Config.fromfile(args.config)
    configs.metric_filename = args.out_name
    configs.training.audio_outputs = f"{configs.vis_dir}_{args.test_epoch}_{args.out_name}/"
    configs.training.losses = [
        dict(type='AmplitudeLoss', args=dict(sample_rate=48000, mask_beginning=512, mask_end=512), loss_weight=0),
        dict(type='SDRloss', args=dict(mask_beginning=512, mask_end=512), loss_weight=0),
        dict(type='PhaseLoss', args=dict(sample_rate=48000, mask_beginning=512, mask_end=512, ignore_below=0.2), loss_weight=0),
        ]
    print("Testing on: " + configs.dataset_test.data_info_file)

    device = 'cuda'
    model = build_model(configs.model)
    model.load(configs.output_dir, device, suffix=args.test_epoch)
    model.eval().to(device)

    # test a subset of 345 mics at a time to avoid out of memory issues
    dataset_chunk = configs.dataset_test
    dataset_chunk.NUM_FORWARD = 23
    mics_all = dataset_chunk.dome_inds
    assert len(mics_all)/dataset_chunk.NUM_FORWARD == int(len(mics_all)/dataset_chunk.NUM_FORWARD)
    N = int(len(mics_all)/dataset_chunk.NUM_FORWARD)
    errors = []
    for mic_chunk in range(N):
        print('Testing mics subset: ' + str(mic_chunk+1) + '/' + str(N))
        dataset_chunk.dome_inds = mics_all[mic_chunk*dataset_chunk.NUM_FORWARD:(mic_chunk+1)*dataset_chunk.NUM_FORWARD]
        dataset_test = build_dataset(dataset_chunk)
        dataset = {'test': dataset_test, 'train': None, 'val': None}
        training_configs = copy.deepcopy(configs)
        trainer = TrainerDp(training_configs.training, model, dataset, seed=args.seed, do_log=False)
        err = trainer.test(save=args.save, rescale=True)
        errors.append(err)
    mean_error = {}
    for metric in errors[0]:
        mean_error[metric] = sum([err[metric].item() for err in errors]) / N

    # print accumulated errors on testset
    result_log = [
        f"{args.config.split('/')[-1][:-3]}, {args.test_epoch}",
        str(configs.test_info_file), '\n',
    ]
    for metric in mean_error:
        log = [metric + " : " +  str(mean_error[metric]) + '\n']
        result_log.extend(log)
    for line in result_log:
        print(line)
    os.makedirs(configs.training.audio_outputs, exist_ok=True)
    with open(f"{configs.training.audio_outputs}/0_metrics.txt", "a") as f:
        f.writelines('\n'.join(result_log))
