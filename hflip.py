from os.path import join as opj

from PIL import Image
import torchvision.transforms.functional as F
import pandas as pd


base_dir = '/home/rlan/projects/self-driving-car-engineer/CarND-Behavioral-Cloning-P3/sharp-turn'
img_src_dir = opj(base_dir, 'IMG')
driving_log_path = opj(base_dir, 'driving_log.csv')


if __name__ == '__main__':
    driving_log = pd.read_csv(driving_log_path, names=['c_image_path', 'l_image_path', 'r_image_path', 'steer',
                                                       'throttle', 'break', 'speed'])
    driving_log_flipped = driving_log.copy()
    for idx, (c, l, r, s, th, br, sp) in driving_log.iterrows():
        for p, col in zip([c, l, r], ['c_image_path', 'l_image_path', 'r_image_path']):
            p_new = p.replace('.jpg', '_flipped.jpg')
            F.hflip(Image.open(p)).save(p_new, 'JPEG')
            driving_log_flipped.loc[idx, col] = p_new

        driving_log_flipped.loc[idx, 'steer'] = -driving_log_flipped.loc[idx, 'steer']

    driving_log_flipped.to_csv(driving_log_path.replace('.csv', '_flipped.csv'), index=False, header=False)
