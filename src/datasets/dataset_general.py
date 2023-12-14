"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os.path as osp
import torch as th
import torchaudio as ta
import numpy as np
import random
import json
from .builder import DATASETS


@DATASETS.register_module()
class AudioDataset:
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mic_loc_file,
                 chunk_size_ms=200,
                 NUM_FORWARD=8,
                 NUM_OUTPUT=345,
                 hmc_inds=[],
                 dome_inds=[],
                 ):
        super().__init__()

        self.data_dir = data_dir
        self.audio_sr = 48000
        self.pose_fps = 30
        self.ap_fps_ratio = self.audio_sr // self.pose_fps
        self.chunk_size = chunk_size_ms * 48
        self.NUM_FORWARD = NUM_FORWARD
        self.NUM_OUTPUT = NUM_OUTPUT
        self.NUM_HMC = 7
        self.NUM_DOME_NODE = 6
        self.NUM_MICS_PER_NODE = 58
        assert self.NUM_DOME_NODE * self.NUM_MICS_PER_NODE >= NUM_OUTPUT
        self.mic_locs = np.load(mic_loc_file)
        if len(hmc_inds) == 0:
            self.hmc_inds = list(range(self.NUM_HMC))
        else:
            self.hmc_inds = hmc_inds
        
        if len(dome_inds) == 0:
            self.dome_indices = list(range(NUM_OUTPUT))
        else:
            self.dome_indices = dome_inds

        self.hmc_fname = lambda idx: f"hmc_audio/hmc_mic{idx:02d}.wav"
        self.dome_fname = lambda node_idx, mic_idx: f"dome_audio/dome_node{node_idx:02d}_mic{mic_idx:02d}.wav"

        with open(data_info_file, 'r') as f:
            self.data_info = json.load(f)
        self.chunks = []
        for date, subdict in self.data_info.items():
            for video, starts in subdict.items():
                curr_chunks = [(date, video, start) for start in starts]
                self.chunks.extend(curr_chunks)

    def __len__(self):
        '''
        :return: number of training chunks in dataset
        '''
        return len(self.chunks)
    
    def _load_hmc_data(self, audio_folder, offset):
        hmc_audio = []
        for i in self.hmc_inds:
            audio_file = osp.join(audio_folder, self.hmc_fname(i+1))
            curr_channel, _ = ta.load(audio_file, frame_offset=offset, num_frames=self.chunk_size)
            hmc_audio.append(curr_channel)

        return th.cat(hmc_audio, dim=0)
    
    def _load_dome_data(self, audio_folder, offset, indices):
        dome_audio = []
        for i in indices:
            node_idx = i // self.NUM_MICS_PER_NODE + 1
            mic_idx = i % self.NUM_MICS_PER_NODE + 1
            audio_file = osp.join(audio_folder, self.dome_fname(node_idx, mic_idx))
            curr_channel, _ = ta.load(audio_file, frame_offset=offset, num_frames=self.chunk_size)
            dome_audio.append(curr_channel)
        return th.cat(dome_audio, dim=0)

    def __getitem__(self, idx):
        info = self.chunks[idx]
        date, video, offset = info
        audio_folder = osp.join(self.data_dir, date, video)
        hmc_audio = self._load_hmc_data(audio_folder, offset)
        indices = random.sample(self.dome_indices, self.NUM_FORWARD)
        dome_audio = self._load_dome_data(audio_folder, offset, indices)
        info = [info, indices]
        mics = self.mic_locs[indices]

        return hmc_audio, dome_audio, 0, mics, 0, info


@DATASETS.register_module()
class AudioHMCDataset(AudioDataset):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mic_loc_file,
                 chunk_size_ms=200,
                 NUM_FORWARD=8,
                 NUM_OUTPUT=345,
                 hmc_inds=[],
                 dome_inds=[],
                 shift_dist_keypoints=[],
                 ):
        super().__init__(data_dir, data_info_file, mic_loc_file, chunk_size_ms, NUM_FORWARD,
            NUM_OUTPUT, hmc_inds, dome_inds)

        self.shift_dist_keypoints = shift_dist_keypoints
        self.all_hmc = {}
        self.shift_dist = {}
        for date, subdict in self.data_info.items():
            self.all_hmc[date] = {}
            self.shift_dist[date] = {}
            for video in subdict:
                curr_hmc = []
                for i in self.hmc_inds:
                    hmc_channel, _ = ta.load(osp.join(self.data_dir, date, video, self.hmc_fname(i+1)))
                    curr_hmc.append(hmc_channel)
                self.all_hmc[date][video] = th.cat(curr_hmc, dim=0)

    def __getitem__(self, idx):
        info = self.chunks[idx]
        date, video, offset = info
        audio_folder = osp.join(self.data_dir, date, video)
        hmc_audio = self.all_hmc[date][video][:, offset: offset + self.chunk_size]
        indices = random.sample(self.dome_indices, self.NUM_FORWARD)
        dome_audio = self._load_dome_data(audio_folder, offset, indices)
        if len(self.shift_dist_keypoints) > 0:
            curr_shift_dist = np.zeros((self.chunk_size // self.ap_fps_ratio, len(indices), len(self.shift_dist_keypoints)), dtype=np.float32)
        else:
            curr_shift_dist = np.zeros((self.chunk_size//self.ap_fps_ratio, len(indices)), dtype=np.float32)
        info = [info, indices]
        mics = self.mic_locs[indices]

        # the amplitude of speech is quite small
        if 'conversation' in video or 'rainbow' in video:
            return hmc_audio*10, dome_audio*10, 0, mics, curr_shift_dist, info
        else:
            return hmc_audio, dome_audio, 0, mics, curr_shift_dist, info


@DATASETS.register_module()
class AudioHMCPoseDataset(AudioHMCDataset):
    def __init__(self,
                 data_dir,
                 data_info_file,
                 mic_loc_file,
                 pose_data_dir=None,
                 chunk_size_ms=200,
                 pose_len=100,
                 NUM_FORWARD=8,
                 NUM_OUTPUT=345,
                 hmc_inds=[],
                 dome_inds=[],
                 keypoint_ids=[],
                 pose_normalize='direct',
                 shift_dist_keypoints=[],
                 ):
        super().__init__(data_dir, data_info_file, mic_loc_file, chunk_size_ms, NUM_FORWARD,
            NUM_OUTPUT, hmc_inds, dome_inds, shift_dist_keypoints)

        self.pose_len = pose_len
        self.pose_normalize = pose_normalize
        assert keypoint_ids[:5] == [0, 15, 16, 17, 18]  # ensure the front five are head pose
        self.pose_mean = np.array([84.02447,   83.84457, 1421.2708], dtype=np.float32)
        self.pose_std = np.array([500., 500., 500.], dtype=np.float32)
        self.pose = {}
        if pose_data_dir is None:
            pose_data_dir = data_dir
        print("pose data dir:", pose_data_dir)
        for date, subdict in self.data_info.items():
            self.pose[date] = {}
            for video in subdict:
                curr_pose = np.load(f"{pose_data_dir}/{date}/{video}/keypoints/keypoints_3d.npy")
                curr_pose = curr_pose.astype(np.float32)  # shape: [Len, Num, 3/7]
                if len(keypoint_ids) > 0:
                    curr_pose = curr_pose[:, keypoint_ids]
                self.pose[date][video] = curr_pose[..., :3]

    def _pose_processing(self, pose):
        if self.pose_normalize == 'direct':
            pose /= 1000.
        elif self.pose_normalize == 'mean_std':
            pose = (pose - self.pose_mean) / (self.pose_std + 1e-4)
        else:
            raise ValueError("please input correct normalization type")
        
        return pose

    def __getitem__(self, idx):
        hmc_audio, dome_audio, _, mics, curr_shift_dist, info = super().__getitem__(idx)
        date, video, offset = self.chunks[idx]
        curr_pose = self.pose[date][video]
        pose_end = (offset + self.chunk_size) // self.ap_fps_ratio
        if pose_end < self.pose_len:
            pose = curr_pose[:pose_end]
            pose_padding = np.stack([pose[0]] * (self.pose_len - pose_end), axis=0)
            pose = np.concatenate([pose_padding, pose], axis=0)
        else:
            pose = curr_pose[pose_end - self.pose_len: pose_end]
        if len(self.shift_dist_keypoints) > 0:
            pose_frames = self.chunk_size // self.ap_fps_ratio
            mic_pos = np.expand_dims(mics, axis=0)
            mic_pos = np.broadcast_to(mic_pos, (pose_frames, mics.shape[0], mics.shape[1]))
            hmc_pos = np.broadcast_to(pose[-pose_frames:, 0:1, :], (pose_frames, mics.shape[0], mics.shape[1]))
            hmc_pos = hmc_pos / 1000
            mic_pos = mic_pos / 1000
            for shift_k in range(len(self.shift_dist_keypoints)):
                if self.shift_dist_keypoints[shift_k] != -1:
                    keypoint_pos = np.broadcast_to(pose[-pose_frames:, self.shift_dist_keypoints[shift_k]: self.shift_dist_keypoints[shift_k] + 1, :],
                                                   (pose_frames, mics.shape[0], mics.shape[1]))
                    keypoint_pos = keypoint_pos / 1000
                    curr_shift_dist[..., shift_k] = np.linalg.norm(keypoint_pos - mic_pos, axis=2) - np.linalg.norm(keypoint_pos - hmc_pos, axis=2)
            curr_shift_dist = - self.audio_sr * curr_shift_dist / 343.0
        pose = self._pose_processing(pose)
        mics = self._pose_processing(mics)

        return hmc_audio, dome_audio, pose, mics, curr_shift_dist, info
