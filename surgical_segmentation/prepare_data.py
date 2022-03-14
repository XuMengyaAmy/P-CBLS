"""
[1] Merge masks with different instruments into one binary mask
[2] Crop black borders from images and masks
"""
from pathlib import Path
import sys
from tqdm import tqdm
import cv2
import numpy as np

data_path = Path('data')

train_path = data_path / '2018_original' / 'train'
val_path = data_path / '2018_original' / 'val'
train_val_path = [train_path, val_path]

original_height, original_width = 1080, 1920
height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instrument_factor = 32

def get_tors_images():
    target_tors_seqs = [1, 5, 7, 9, 11, 19, 20, 21]
    target_images = []
    for seq in target_tors_seqs:
        target_images += list((data_path / 'tors' / 'images' / str(seq)).glob('*'))
    for i in range(len(target_images)):
        target_images[i] = str(target_images[i])
    return target_images

def get_2017_images():
    target_images = []
    target_images += list((data_path / '2017_selected' / 'images').glob('*'))
    for i in range(len(target_images)):
        target_images[i] = str(target_images[i])
    return target_images

def get_2017_images_cropped():
    target_images = []
    target_images += list((data_path / '2017_selected' / 'images_cropped').glob('*'))
    for i in range(len(target_images)):
        target_images[i] = str(target_images[i])
    return target_images
    

if __name__ == '__main__':
    for path in train_val_path:
        instrument_mask_folder = (path / 'instruments_masks')
        instrument_mask_folder.mkdir(exist_ok=True, parents=True)

        for filename in tqdm(list((path / 'annotations').glob('*'))):
            # mask_instruments = np.zeros((height, width))
            mask_instruments = cv2.imread(str(filename), 0)

            mask_instruments = mask_instruments.astype(np.uint8) * instrument_factor

            cv2.imwrite(str(instrument_mask_folder / filename.name), mask_instruments)
