"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

data_dir = '/dataset/'
mic_loc_file = '/dataset/mic_loc_3D.npy'
exp_type = 'neurips'
exp_name = 'test_run'
output_dir = f'./checkpoint/{exp_type}/{exp_name}'

# training
training = dict(
    output_dir=output_dir,
    num_gpus=4,
    batch_size_per_gpu=1,
    save_frequency=4,
    val_frequency=4,
    epochs=120,
    optim=dict(type='AdamW', init_lr=2e-3),
    losses=[
        dict(type='ShiftedL2Loss', args=dict(sample_rate=48000, mask_beginning=512, mask_end=512), loss_weight=1),
        dict(type='MultiResolutionSTFTLoss', args=dict(sample_rate=48000, mask_beginning=512, mask_end=512,
                                                       fft_bins_list=[256, 128, 64, 32]),
             name='multiSTFTloss', loss_weight=100),
    ],
    weight_clip=500,
    max_amplify=50,                                            # dynamic linear scaling
    # start_epoch=30
)

# dataset
dataset_base = dict(
    type='AudioHMCPoseDataset',
    data_dir=data_dir,
    mic_loc_file=mic_loc_file,
    chunk_size_ms=1000,
    pose_len=38,
    NUM_FORWARD=38,
    hmc_inds=[0, 1, 2, 3, 4, 5, 6],
    dome_inds=list(set(range(0, 348)) - {249, 250, 251}),   # remove empty channels (node 5 mics 18, 19, 20)
    keypoint_ids=[0, 15, 16, 17, 18] + list(range(1, 15)),  # ensure the front five are head pose (moves other kp by +4)
    shift_dist_keypoints=[-1, 0, 8, 11, 12, 15, 18],        # [no shift, kp0, kp4 +4, kp7 +4, kp8 +4, kp11 +4, kp14 +4]
    pose_normalize='mean_std',
)

# model
model = dict(
    type='AudioPoseModel',
    audio_net=dict(
        type='AudioHeadPoseNet',
        input_channels=49,
        mic_embed_dim=4,
        head_embed_dim=8,
        pose_embed_dim=32,
        wavenet_blocks=1,
        layers_per_block=10,
        wavenet_channels=128,
        conv_kernel=3,
        causal=False,
        convblock_name='ConvBlock',
    ),
    pose_net=dict(
        type='PoseConv2DNet',
        in_channels=3,
        base_channels=32,
        out_channels=32,
        num_keypoints=19,
        kernel_size=5,
    ),
    lr=dict(audio=2e-3, pose=2e-3),
    pre_pose_len=8,
)

# dataset_train = dict(data_info_file='./data_info/train/all_data.json')
# dataset_val = dict(data_info_file='./data_info/val/all_data.json')
# removed segments containing only silence (silent threshold on headset mics = 2 * 1e-3)
dataset_train = dict(data_info_file='./data_info/train/all_data_no_silence.json')
dataset_val = dict(data_info_file='./data_info/val/all_data_no_silence.json')
dataset_train.update(dataset_base)
dataset_val.update(dataset_base)

# testing
vis_dir = f'./outputs/{exp_type}/{exp_name}'
test_info_file = './data_info/test/nonspeech_data.json'
# test_info_file = './data_info/test/speech_data.json'
dataset_test = dict(data_info_file=test_info_file)
dataset_test.update(dataset_base)
