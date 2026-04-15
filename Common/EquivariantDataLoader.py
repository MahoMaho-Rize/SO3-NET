"""
PyTorch Geometric data loader for equivariant upright orientation estimation.

Both train and test load only the original objects (1110 train / 370 test).
Random SO(3) rotations are applied on-the-fly every time a sample is accessed,
providing infinite augmentation without pre-generated rotation files.
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from scipy.spatial.transform import Rotation


class EquivariantPointCloudDataset(Dataset):
    """
    Point cloud dataset with on-the-fly random rotation.

    - Train: 1110 objects, random rotation each access
    - Test:  370 objects, random rotation each access

    Each sample returns:
        data.pos:          (N, 3) randomly rotated point cloud
        data.pos_original: (N, 3) canonical pose
        data.y_direction:  (1, 3) GT upright direction = R[:, 1]
        data.y_support:    (N,)   GT support labels
        data.rotm:         (3, 3) rotation matrix applied
        data.label:        (1,)   category label
    """

    def __init__(self, opts, partition="test"):
        super().__init__()
        self.num_points = opts.num_points
        data_dir = opts.data_dir

        self.data_original = np.load(
            os.path.join(data_dir, "%s_noaug_original.npy" % partition)
        ).astype(np.float32)
        self.labels = np.load(
            os.path.join(data_dir, "labels_%s_noaug.npy" % partition)
        ).astype(np.float32)
        self.pid = np.load(
            os.path.join(data_dir, "pid_%s_noaug.npy" % partition)
        ).astype(np.float32)

    def len(self):
        return len(self.data_original)

    def get(self, index):
        original = self.data_original[index][: self.num_points, :].copy()
        support_labels = self.pid[index][: self.num_points].copy()
        label = self.labels[index].copy()

        original = torch.from_numpy(original).float()
        support_labels = torch.from_numpy(support_labels).float()
        label = torch.tensor(label).long()

        # On-the-fly random rotation
        R = torch.from_numpy(Rotation.random().as_matrix().astype(np.float32))
        pos = (R @ original.T).T
        y_direction = R[:, 1].unsqueeze(0)  # (1, 3) for PyG batching

        return Data(
            pos=pos,
            pos_original=original,
            y_direction=y_direction,
            y_support=support_labels,
            rotm=R,
            label=label,
        )
