import numpy as np
import warnings
from torch.utils.data import Dataset
import os
warnings.filterwarnings('ignore')

class RobustPointSetDataLoader(Dataset):
    def __init__(self, opts, partition='test'):
        self.opts = opts
        self.data_original = np.load(os.path.join(opts.data_dir, '%s_original.npy'%partition))
        self.data_rotation = np.load(os.path.join(opts.data_dir, '%s_rotation.npy'%partition))
        self.labels = np.load(os.path.join(opts.data_dir, 'labels_%s.npy'%partition))
        self.rotm = np.load(os.path.join(opts.data_dir, 'rotm_%s.npy'%partition))
        self.pid = np.load(os.path.join(opts.data_dir, 'pid_%s.npy'%partition))
        self.coef_d = np.load(os.path.join(opts.data_dir, '%s_d.npy'%partition))
        self.num_points = opts.num_points
        self.partition = partition

    def __len__(self):
        return len(self.data_rotation)

    def __getitem__(self, index):
        data_original = self.data_original[index][:self.num_points,:].copy()
        data_rotation = self.data_rotation[index][:self.num_points,:].copy()
        labels = self.labels[index]
        rotm = self.rotm[index]
        pid = self.pid[index][:self.num_points].copy()
        coef_d = self.coef_d[index]
        return data_original.astype(np.float32), data_rotation.astype(np.float32), labels.astype(np.int32), \
               rotm.astype(np.float32), pid.astype(np.int32), coef_d.astype(np.float32)

