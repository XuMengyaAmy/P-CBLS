from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from prepare_data import height, width, h_start, w_start, instrument_factor


def general_dice(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [dice(y_true == instrument_id, y_pred == instrument_id)]

        class_id = int(instrument_id / instrument_factor)
        result_dice_each[classes_2018[class_id - 1]] += [dice(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def general_jaccard(y_true, y_pred):
    result = []

    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    for instrument_id in set(y_true.flatten()):
        if instrument_id == 0:
            continue
        result += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

        #### show results for each instru
        class_id = int(instrument_id / instrument_factor)
        result_jaccard_each[classes_2018[class_id - 1]] += [jaccard(y_true == instrument_id, y_pred == instrument_id)]

    return np.mean(result)


def jaccard(y_true, y_pred):
    # print('y_true', y_true) #  [False False False ... False False False]
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

# https://blog.csdn.net/lingzhou33/article/details/87901365
# 实现代码也很简单：intersection = np.logical_and(target, prediction) union = np.logical_or(target, prediction) iou_score = np.sum(intersection) / np.sum(union)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='data/2018_original',
        help='path where train images with ground truth are located')
    
    arg('--gt_path', type=str, default='data/2018_original/val',
        help='path where images with ground truth are located')   
    arg('--target_path', type=str, default='predictions/UNet11', help='path with predictions')
    arg('--problem_type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    args = parser.parse_args()

    result_dice = []
    result_jaccard = []

    result_dice_each = {}
    result_jaccard_each = {}

    classes_2018 = ['Bipolar_Forceps', 'Prograsp_Forceps', 'Large_Needle_Driver',
                    'Monopolar_Curved_Scissors', 'Ultrasound_Probe', 'Suction_Instrument', 'Clip_Applier']

    # classes_2017 = ['Bipolar_Forceps', 'Prograsp_Forceps', 'Large_Needle_Driver',
    #                 'Vessel_Sealer', 'Grasping_Retractor', 'Monopolar_Curved_Scissors', 'Other']

    for i in classes_2018:
        result_dice_each[i] = []
        result_jaccard_each[i] = []
    
    if args.problem_type == 'binary':
        for instrument_id in tqdm(range(1, 9)):
            instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)

            for file_name in (
                    Path(args.train_path) / instrument_dataset_name / 'binary_masks').glob('*'):
                y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

                pred_file_name = (Path(args.target_path) / 'binary' / instrument_dataset_name / file_name.name)

                pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * 0.5).astype(np.uint8)
                y_pred = pred_image[h_start:h_start + height, w_start:w_start + width]

                result_dice += [dice(y_true, y_pred)]
                result_jaccard += [jaccard(y_true, y_pred)]

    elif args.problem_type == 'parts':
        for instrument_id in tqdm(range(1, 9)):
            instrument_dataset_name = 'instrument_dataset_' + str(instrument_id)
            for file_name in (
                    Path(args.train_path) / instrument_dataset_name / 'parts_masks').glob('*'):
                y_true = cv2.imread(str(file_name), 0)

                pred_file_name = Path(args.target_path) / 'parts' / instrument_dataset_name / file_name.name

                y_pred = cv2.imread(str(pred_file_name), 0)[h_start:h_start + height, w_start:w_start + width]

                result_dice += [general_dice(y_true, y_pred)]
                result_jaccard += [general_jaccard(y_true, y_pred)]

    elif args.problem_type == 'instruments':
        for file_name in (
                Path(args.gt_path) / 'instruments_masks').glob('*'): # should only use val data to evaluate
            y_true = cv2.imread(str(file_name), 0)　# flag=0, read the gray image

            pred_file_name = Path(args.target_path) / file_name.name

            y_pred = cv2.imread(str(pred_file_name), 0)

            result_dice += [general_dice(y_true, y_pred)]
            result_jaccard += [general_jaccard(y_true, y_pred)]
    
    for i in classes_2018:
        if result_dice_each[i] == []:
            result_dice_each[i] = [0]
        if result_jaccard_each[i] == []:
            result_jaccard_each[i] = [0]

    print('Model: ', args.target_path)
    # print('Mean Dice = ', np.around(np.mean(result_dice), 6))
    # for instrument in classes_2018:
    #     print('Dice of {} is {}.'.format(str(instrument), np.around(np.mean(result_dice_each[instrument]), 6)))
    print('Mean IOU = ', np.around(np.mean(result_jaccard), 6))
    for instrument in classes_2018:
        print('IOU of {} is {}.'.format(str(instrument), np.around(np.mean(result_jaccard_each[instrument]), 6)))

