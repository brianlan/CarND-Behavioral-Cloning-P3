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

        self.indices.loc[self.indices.img_type == 'r_image_path', 'steer'] -= 0.08
        self.indices.loc[self.indices.img_type == 'l_image_path', 'steer'] += 0.08

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
