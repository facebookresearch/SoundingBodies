"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import matplotlib.pyplot as plt


skeleton = [
    (0, 1),
    (0, 15),
    (0, 16),
    (1, 2),
    (1, 5),
    (1, 8),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (8, 12),
    (9, 10),
    (10, 11),
    (11, 22),
    (11, 24),
    (12, 13),
    (13, 14),
    (14, 19),
    (14, 21),
    (15, 17),
    (16, 18),
    (19, 20),
    (22, 23),
]


def cam2ane(x):
    R = np.matrix([[-9.48549486e-04, 2.28584263e-01, -9.73523669e-01],
                   [9.96460005e-01, 8.20532798e-02, 1.82952890e-02],
                   [8.40628252e-02, -9.70060047e-01, -2.27852907e-01]])
    t = np.matrix([974.60911285, 40.79278411, 1622.38139632]) / 1000

    y = np.matmul(x - t, R)
    return y


def ane2cam(x):
    R = np.matrix([[-9.48549486e-04, 2.28584263e-01, -9.73523669e-01],
                   [9.96460005e-01, 8.20532798e-02, 1.82952890e-02],
                   [8.40628252e-02, -9.70060047e-01, -2.27852907e-01]])
    t = np.matrix([974.60911285, 40.79278411, 1622.38139632]) / 1000

    y = np.matmul(x, np.linalg.inv(R)) + t
    return y


if __name__ == "__main__":

    mic_loc_file = 'mic_loc_3D.npy'
    keypoints_3d_file = '../data_tmp/keypoints_3d.npy'

    mic_loc = np.load(mic_loc_file)
    keypoints_3d = np.load(keypoints_3d_file)

    dome_inds = list(set(range(0, 348)) - {249, 250, 251})
    frame = 0
    keypoints_3d = keypoints_3d[frame, :, :3] / 1000
    mic_loc = mic_loc[dome_inds, :] / 1000

    keypoints_3d = cam2ane(keypoints_3d)
    mic_loc = cam2ane(mic_loc)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mic_loc[:, 0], mic_loc[:, 1], mic_loc[:, 2], '.r')
    ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2], 'b')
    lines = [keypoints_3d[edge, :] for edge in skeleton]
    for i, line in enumerate(lines):
        ax.plot3D([line[0, 0], line[1, 0]], [line[0, 1], line[1, 1]], [line[0, 2], line[1, 2]], 'b')
    plt.show()
