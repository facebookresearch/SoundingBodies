"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
import os.path as osp
from datetime import datetime
from collections import OrderedDict
import tqdm
import time
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
import random
import argparse
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from .utils import *
from .losses import *


class Trainer:
    def __init__(self, config, model, dataset, seed, do_log=True):
        """
        :param config: a dict containing parameters
        :param model: the model to be trained
        :param dataset: includes both training and validation dataset
        """
        config["batch_size"] = config["num_gpus"] * config["batch_size_per_gpu"]
        self.config = config
        self.do_log = do_log
        self.model = None
        self.optimizer = None
        self.dataloader_train = None
        self.sampler_train = None
        self.dataloader_val = None
        self.sampler_val = None
        self.dataloader_test = None
        self.sampler_test = None
        self.init_dataloaders(dataset, seed)

        # losses
        self.losses = config.losses.copy()
        for loss in self.losses:
            loss_class = globals()[loss['type']]
            loss['name'] = loss.get('name', loss['type'].lower())
            if 'args' in loss:
                loss['loss'] = loss_class(**loss['args'])
            else:
                loss['loss'] = loss_class()

        # resume training
        self.total_iters = 0
        self.start_epoch = config.get('start_epoch', 0)
        if self.start_epoch > 0:
            resume_epoch = f'epoch-{self.start_epoch}'
            if "resume_dir" in config:
                resume_dir = config["resume_dir"]
            else:
                resume_dir = config["output_dir"]
            assert osp.exists(resume_dir)
            model.load(
                resume_dir,
                suffix=resume_epoch,
                load_audio_only=config.get("load_audio_only", False),
                strict=config.get("load_strict", True),
            )
            self.optimizer_state_dict = th.load(osp.join(resume_dir, f'optim-{resume_epoch}.pth'), map_location='cuda')
            self.total_iters = self.optimizer_state_dict.get("total_iters", 0)

        self.setup_logging()
        self.setup_optimizer(model)
        self.model.train()

    def init_dataloaders(self, dataset, seed):
        super().init_dataloaders(dataset, seed)

    def setup_optimizer(self, model):
        super().setup_optimizer(model)

    def setup_logging(self):
        if self.do_log and is_main_process():
            self.logger = SummaryWriter(
                osp.join(self.config["output_dir"], 'logs', datetime.now().strftime("%Y%m%d-%H%M%S")))

    def log(self, name, value, epoch):
        if self.do_log and is_main_process():
            self.logger.add_scalar(name, value, epoch)

    def print(self, text):
        if is_main_process():
            print(text)

    def save(self, suffix=""):
        if is_main_process():
            self.model.module.save(self.config["output_dir"], suffix)
            optimizer_state_dict = {'optimizer': self.optimizer.state_dict(), 'total_iters': self.total_iters}
            th.save(optimizer_state_dict, osp.join(self.config["output_dir"], f'optim-{suffix}.pth'))

    @staticmethod
    def _parse_losses(losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, th.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        return loss

    def forward_step(self, data, save=False, rescale=False, audio_sr=48000):
        # data processing
        hmc_audio, dome_audio, pose, mics, shift_dist, info = data
        hmc_audio, dome_audio, pose = hmc_audio.cuda(), dome_audio.cuda(), pose.cuda()
        B = hmc_audio.shape[0]
        hmc_amplify = 1 / th.abs(hmc_audio.reshape(B, -1)).max(dim=1)[0]
        dome_amplify = 1 / th.abs(dome_audio.reshape(B, -1)).max(dim=1)[0]
        final_amplify = th.minimum(hmc_amplify, dome_amplify).view(B, 1, 1)
        final_amplify = final_amplify.clamp(max=self.config['max_amplify'])
        hmc_audio *= final_amplify
        dome_audio *= final_amplify

        # forward
        prediction = self.model.forward(hmc_audio, pose, mics, shift_dist)
        pred_dome = prediction['output']
        if rescale:
            hmc_audio /= final_amplify
            dome_audio /= final_amplify
            pred_dome /= final_amplify
        if save:
            os.makedirs(self.config["audio_outputs"], exist_ok=True)
            for i in range(dome_audio.shape[0]):
                t_sec = info[0][2][i] / audio_sr
                if int(t_sec) == t_sec:
                    t_sec = int(t_sec)
                for j in range(dome_audio.shape[1]):
                    out_name = info[0][1][i] + '_' + str(t_sec) + '_' + str(info[1][j][i].item()) + '.wav'
                    out_wav = th.stack([hmc_audio[i, 0, :], dome_audio[i, j, :], pred_dome[i, j, :]], dim=0).cpu()
                    ta.save(osp.join(self.config["audio_outputs"], out_name), out_wav, audio_sr)

        # observe loss
        loss_results = {}
        loss_obs = {}
        for loss in self.losses:
            output_loss = loss['loss'](pred_dome, dome_audio)
            loss_results[loss['name'].replace('loss', '')] = output_loss
            loss_results[loss['name']] = output_loss * loss['loss_weight']
        for k, v in loss_results.items():
            if 'loss' not in k:
                loss_obs[k] = v.detach()
        total_loss = self._parse_losses(loss_results)
        loss_obs['accumulated_loss'] = total_loss.detach()

        if is_dist_avail_and_initialized():
            dist.barrier()
        return loss_obs, total_loss

    def train(self):
        val_best_epoch = {}
        for epoch in range(self.start_epoch, self.config["epochs"]):
            epoch += 1
            if self.sampler_train is not None:
                self.sampler_train.set_epoch(epoch)

            loss_stats = {}
            t_start = time.time()
            data_pbar = tqdm.tqdm(self.dataloader_train)
            for data in data_pbar:
                self.total_iters += 1
                self.optimizer.zero_grad()
                loss_obs, total_loss = self.forward_step(data)
                if total_loss is None or self.config.get('dummy_run', False):
                    pass
                else:
                    total_loss.backward()
                    if 'weight_clip' in self.config and epoch > self.config.get('clip_epoch', 0):
                        total_norm = th.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['weight_clip'],
                                                                 error_if_nonfinite=True)
                        self.log(f'Loss/grad_norm', total_norm.cpu(), self.total_iters)
                    self.optimizer.step()
                # logging
                for k, v in loss_obs.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
                    self.log(f'Loss/{k}', v.data.cpu(), self.total_iters)
                data_pbar.set_description(f"loss: {loss_obs['accumulated_loss'].item():.7f}")
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader_train)
                self.log(f'Avg_loss/{k}', loss_stats[k].detach().data.cpu(), epoch)
            self.optimizer.update_lr(self.model, loss_stats["accumulated_loss"])
            t_end = time.time()
            loss_str = "    ".join([f"{k}:{v.item():.4}" for k, v in loss_stats.items()])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            self.print(f"epoch {epoch} " + loss_str + "        " + time_str)

            # save model
            if self.config["save_frequency"] > 0 and epoch % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch))
                self.print("Saved model")
            # validate
            if self.config["val_frequency"] > 0 and epoch % self.config["val_frequency"] == 0:
                self.print(f"==== Validate on epoch {epoch} ====")
                curr_val_loss = self.validate(epoch)
                self.print(curr_val_loss)
                self.print(f"=============================")
                for k, v in curr_val_loss.items():
                    if k not in val_best_epoch or val_best_epoch[k][1] > v:
                        val_best_epoch[k] = (epoch, v)
                        self.save(suffix=f'best-{k}')

        # save final model
        self.save(suffix="final")
        if is_main_process():
            with open(osp.join(self.config["output_dir"], 'train_info.txt'), 'a') as f:
                for k, v in val_best_epoch.items():
                    f.write(f"{k}: epoch {v[0]}, value {v[1]}\n")
            if self.do_log:
                self.logger.close()
        if is_dist_avail_and_initialized():
            cleanup()

    def validate(self, epoch):
        loss_val_stats = {}
        n = 0
        for data in self.dataloader_val:
            n += 1
            with th.no_grad():
                loss_obs, _ = self.forward_step(data)
                for k, v in loss_obs.items():
                    loss_val_stats[k] = loss_val_stats[k]+v if k in loss_val_stats else v
        for k in loss_val_stats:
            loss_val_stats[k] /= n
            self.log(f'Val_loss/{k}', loss_val_stats[k].data.cpu(), epoch)
        return loss_val_stats

    def test(self, save=True, rescale=True):
        loss_test_stats = {}
        n = 0
        for data in self.dataloader_test:
            n += 1
            with th.no_grad():
                loss_obs, _ = self.forward_step(data, save=save, rescale=rescale)
                for k, v in loss_obs.items():
                    loss_test_stats[k] = loss_test_stats[k]+v if k in loss_test_stats else v
        for k in loss_test_stats:
            loss_test_stats[k] /= n
        return loss_test_stats


class TrainerDp(Trainer):
    def __init__(self, config, model, dataset, seed, do_log=True):
        super().__init__(config, model, dataset, seed, do_log)

    def init_dataloaders(self, dataset, seed):
        def seed_worker(seed):
            np.random.seed(seed)
            random.seed(seed)
        g = th.Generator()
        g.manual_seed(seed)
        if dataset['train'] is not None:
            self.dataloader_train = DataLoader(
                dataset['train'],
                batch_size=self.config["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=20,
                worker_init_fn=seed_worker,
                generator=g,
            )
        if dataset['val'] is not None:
            self.dataloader_val = DataLoader(
                dataset['val'],
                batch_size=self.config["batch_size"],
                shuffle=False,
                drop_last=True,
                num_workers=20,
                worker_init_fn=seed_worker,
                generator=g,
            )
        if dataset['test'] is not None:
            self.dataloader_test = DataLoader(
                dataset['test'],
                batch_size=self.config["batch_size"],
                shuffle=False,
                drop_last=False,
                num_workers=20,
                worker_init_fn=seed_worker,
                generator=g,
            )

    def setup_optimizer(self, model):
        optim_class = globals()[self.config.optim.pop('type')]
        gpus = [i for i in range(self.config["num_gpus"])]
        self.model = th.nn.DataParallel(model, gpus)
        self.optimizer = optim_class(model.weights, **self.config.optim)
        if self.start_epoch > 0 and not self.config.get("no_load_optim", False):
            self.optimizer.load_state_dict(self.optimizer_state_dict["optimizer"])


class TrainerDdp(Trainer):
    def __init__(self, config, model, dataset, seed, do_log=True):
        super().__init__(config, model, dataset, seed, do_log)

    def init_dataloaders(self, dataset, seed):
        def seed_worker(seed):
            np.random.seed(seed)
            random.seed(seed)
        init_distributed()
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        if dataset['train'] is not None:
            self.sampler_train = DistributedSampler(dataset['train'], shuffle=True)
            self.dataloader_train = DataLoader(
                dataset['train'],
                batch_size=self.config["batch_size_per_gpu"],
                drop_last=False,
                num_workers=10,
                worker_init_fn=seed_worker,
                sampler=self.sampler_train
            )
        if dataset['val'] is not None:
            self.sampler_val = DistributedSampler(dataset['val'], shuffle=False)
            self.dataloader_val = DataLoader(
                dataset['val'],
                batch_size=self.config["batch_size_per_gpu"],
                drop_last=False,
                num_workers=10,
                worker_init_fn=seed_worker,
                sampler=self.sampler_val
            )
        if dataset['test'] is not None:
            self.sampler_test = DistributedSampler(dataset['test'], shuffle=False)
            self.dataloader_test = DataLoader(
                dataset['test'],
                batch_size=self.config["batch_size_per_gpu"],
                drop_last=False,
                num_workers=10,
                worker_init_fn=seed_worker,
                sampler=self.sampler_test
            )

    def setup_optimizer(self, model):
        optim_class = globals()[self.config.optim.pop('type')]
        local_rank = int(os.environ['LOCAL_RANK'])
        # model = model.to(local_rank)
        model = model.cuda()
        self.model = th.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        self.optimizer = optim_class(self.model.parameters(), **self.config.optim)
        if self.start_epoch > 0 and not self.config.get("no_load_optim", False):
            self.optimizer.load_state_dict(self.optimizer_state_dict["optimizer"])
