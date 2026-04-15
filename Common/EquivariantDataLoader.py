"""
PyTorch Geometric data loader for equivariant upright orientation estimation.

Converts the original .npy dataset format to PyG Data objects, adding:
  - Ground truth upright direction (from rotation matrix)
  - Optional: bypass rotation augmentation for equivariant models

Usage:
    from Common.EquivariantDataLoader import EquivariantPointCloudDataset
    from torch_geometric.loader import DataLoader

    dataset = EquivariantPointCloudDataset(opts, partition='train')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class EquivariantPointCloudDataset(Dataset):
    """
    Point cloud dataset for equivariant upright orientation estimation.

    Each sample contains:
        data.pos:          (N, 3) point cloud coordinates (input to the network)
        data.pos_original: (N, 3) canonical pose coordinates (for evaluation)
        data.y_direction:  (3,)   GT upright direction in the input frame
        data.y_support:    (N,)   GT per-point support labels (0 or 1)
        data.rotm:         (3, 3) rotation matrix (for evaluation)
        data.label:        (1,)   object category label

    When use_rotation_aug=True (default):
        - pos = rotated point cloud
        - y_direction = rotm @ [0, 1, 0] = rotm[:, 1]

    When use_rotation_aug=False (equivariant advantage):
        - pos = original (canonical) point cloud
        - y_direction = [0, 1, 0]
        - Equivariance guarantees generalization to arbitrary rotations
    """

    def __init__(self, opts, partition="test"):
        super().__init__()
        self.opts = opts
        self.partition = partition
        self.num_points = opts.num_points
        self.use_rotation_aug = opts.use_rotation_aug

        # Load .npy data (same format as original RobustPointSetDataLoader)
        data_dir = opts.data_dir
        self.data_original = np.load(
            os.path.join(data_dir, "%s_original.npy" % partition)
        )
        self.data_rotation = np.load(
            os.path.join(data_dir, "%s_rotation.npy" % partition)
        )
        self.labels = np.load(os.path.join(data_dir, "labels_%s.npy" % partition))
        self.rotm = np.load(os.path.join(data_dir, "rotm_%s.npy" % partition))
        self.pid = np.load(os.path.join(data_dir, "pid_%s.npy" % partition))

    def len(self):
        return len(self.data_rotation)

    def get(self, index):
        # Point clouds
        original = self.data_original[index][: self.num_points, :].copy()
        rotated = self.data_rotation[index][: self.num_points, :].copy()
        rotm = self.rotm[index].copy()
        support_labels = self.pid[index][: self.num_points].copy()
        label = self.labels[index].copy()

        # Convert to tensors
        original = torch.from_numpy(original).float()  # (N, 3)
        rotated = torch.from_numpy(rotated).float()  # (N, 3)
        rotm = torch.from_numpy(rotm).float()  # (3, 3)
        support_labels = torch.from_numpy(support_labels).float()  # (N,)
        label = torch.tensor(label).long()

        # Determine input point cloud and GT direction
        if self.use_rotation_aug:
            # Use rotated points; GT direction = rotm @ [0, 1, 0]
            pos = rotated
            y_direction = rotm[:, 1]  # Second column of rotation matrix
        else:
            # Use original (canonical) points; GT direction = [0, 1, 0]
            pos = original
            y_direction = torch.tensor([0.0, 1.0, 0.0])

        # Build PyG Data object
        data = Data(
            pos=pos,  # (N, 3) input point cloud
            pos_original=original,  # (N, 3) canonical coordinates
            y_direction=y_direction,  # (3,)   GT upright direction
            y_support=support_labels,  # (N,)   GT support labels
            rotm=rotm,  # (3, 3) rotation matrix
            label=label,  # (1,)   category label
        )
        return data
