import os
from os.path import basename

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


OPJ = os.path.join


class ImageFolder(Dataset):
    def __init__(self, indices_paths, transform=None, target_transform=None):
        def _read_driving_log(path):
            driving_log = pd.read_csv(path, names=['c_image_path', 'l_image_path', 'r_image_path', 'steer',
                                                   'throttle', 'break', 'speed'])
            return driving_log

        self.transform = transform
        self.target_transform = target_transform
        self.indices = pd.concat([_read_driving_log(p) for p in indices_paths])

        ###############################
        # Leverage Right / Left Image
        ###############################
        self.indices = pd.melt(self.indices, value_name='img_path', var_name='img_type',
                               value_vars=['c_image_path', 'l_image_path', 'r_image_path'],
                               id_vars=['steer', 'throttle', 'break', 'speed'])
        self.indices.loc[:, 'steer'] = self.indices.loc[:, 'steer'].astype(np.float64)
        # big_turn = self.indices[np.abs(self.indices.steer) > 0.15]
        # self.indices = self.indices[self.indices.img_type == 'c_image_path']
        # self.indices = self.indices.loc[:len(self.indices) // 2, :]

        # self.indices.to_csv('~/Downloads/indices.csv', index=False)
        # self.indices = pd.concat([self.indices] + [big_turn] * 4)
        self.indices.loc[self.indices.img_type == 'r_image_path', 'steer'] -= 0.05
        self.indices.loc[self.indices.img_type == 'l_image_path', 'steer'] += 0.05
        # self.indices.loc[self.indices.img_type == 'r_image_path', 'throttle'] -= 0.1
        # self.indices.loc[self.indices.img_type == 'l_image_path', 'throttle'] -= 0.1
        # self.indices.loc[self.indices.throttle < 0, 'throttle'] = 0.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        im = Image.open(self.indices.iloc[item].img_path)
        target = np.array([self.indices.iloc[item].steer, self.indices.iloc[item].throttle])

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target
