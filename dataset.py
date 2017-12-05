import os
from os.path import basename

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


OPJ = os.path.join


class ImageFolder(Dataset):
    def __init__(self, data_dir, indices_path, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.indices = pd.read_csv(indices_path, names=['c_image_path', 'l_image_path', 'r_image_path', 'steer',
                                                        'throttle', 'break', 'speed'])

        ###############################
        # Leverage Right / Left Image
        ###############################
        self.indices = pd.melt(self.indices, value_name='img_path', var_name='img_type',
                               value_vars=['c_image_path', 'l_image_path', 'r_image_path'],
                               id_vars=['steer', 'throttle', 'break', 'speed'])

        self.indices.loc[self.indices.img_type == 'r_image_path', 'steer'] -= 0.15
        self.indices.loc[self.indices.img_type == 'l_image_path', 'steer'] += 0.15
        self.indices.loc[self.indices.img_type == 'r_image_path', 'throttle'] -= 0.1
        self.indices.loc[self.indices.img_type == 'l_image_path', 'throttle'] -= 0.1
        self.indices.loc[self.indices.throttle < 0, 'throttle'] = 0.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        im = Image.open(OPJ(self.data_dir, basename(self.indices.iloc[item].img_path)))
        target = np.array([self.indices.iloc[item].steer, self.indices.iloc[item].throttle])

        if self.transform is not None:
            im = self.transform(im)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target
