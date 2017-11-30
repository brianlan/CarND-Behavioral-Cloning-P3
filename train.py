import os

import pandas as pd
import skimage.io as io
from torch.utils.data import Dataset


OPJ = os.path.join


class BehavioralData(Dataset):
    def __init__(self, data_dir, indices_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.indices = pd.read_csv(indices_path, names=['c_image_path', 'l_image_path', 'r_image_path', 'steer',
                                                        'throttle', 'break', 'speed'])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        im = io.imread(OPJ(self.data_dir, self.indices[item].c_image_path))
        label = self.indices[item].steer

        if self.transform is not None:
            im = self.transform(im)

        return im, label
