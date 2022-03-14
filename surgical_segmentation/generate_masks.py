"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split
from dataset import RoboticsDataset
import cv2
from models import UNet16, LinkNet34, UNet11, UNet, AlbuNet
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
from prepare_data import (get_2017_images_cropped, original_height,
                          original_width,
                          h_start, w_start, 
                          get_tors_images,
                          get_2017_images
                          )
from albumentations import Compose, Normalize
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop, 
    #FDA, 
    Resize
)
from validation import validation_multi
from loss import CELoss

import os


def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type='LinkNet34', problem_type='instruments'):
    """

    :param model_path:
    :param model_type: 'UNet', 'UNet16', 'UNet11', 'LinkNet34', 'AlbuNet'
    :param problem_type: 'binary', 'parts', 'instruments'
    :return:
    """
    if problem_type == 'binary':
        num_classes = 1
    elif problem_type == 'parts':
        num_classes = 4
    elif problem_type == 'instruments':
        num_classes = 8

    if model_type == 'UNet16':
        model = UNet16(num_classes=num_classes)
    elif model_type == 'UNet11':
        model = UNet11(num_classes=num_classes)
    elif model_type == 'LinkNet34':
        model = LinkNet34(num_classes=num_classes)
    elif model_type == 'AlbuNet':
        model = AlbuNet(num_classes=num_classes)
    elif model_type == 'UNet':
        model = UNet(num_classes=num_classes)
    
    print('model_path:', model_path)
    state = torch.load(str(model_path))

    # state = {key.replace('module.', ''): value for key, value in state.items()}
    state = {key.replace('module.', ''): value for key, value in state['net'].items()} # original one
    
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size, to_path, problem_type, img_transform):
    loader = DataLoader(
        dataset=RoboticsDataset(from_file_names, transform=img_transform, mode='predict', problem_type=problem_type),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs = model(inputs)

            for i, image_name in enumerate(paths):
                if problem_type == 'binary':
                    factor = prepare_data.binary_factor
                    t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)
                elif problem_type == 'parts':
                    factor = prepare_data.parts_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)
                elif problem_type == 'instruments':
                    factor = prepare_data.instrument_factor
                    t_mask = (outputs[i].data.cpu().numpy().argmax(axis=0) * factor).astype(np.uint8)

                # h, w = t_mask.shape

                # full_mask = np.zeros((original_height, original_width))
                # full_mask[h_start:h_start + h, w_start:w_start + w] = t_mask

                # instrument_folder = Path(paths[i]).parent.parent.name

                (to_path).mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(to_path / (Path(paths[i]).stem + '.png')), t_mask)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='saved_model_70epochs_lr_1e-4/LinkNet34/baseline/', help='path to model folder')
    
    arg('--model_type', type=str, default='LinkNet34', help='network architecture',
        choices=['UNet', 'UNet11', 'UNet16', 'LinkNet34', 'AlbuNet'])
    arg('--output_path', type=str, help='path to save images', default='1')
    arg('--batch-size', type=int, default=16)
    arg('--num_classes', type=int, default=8)
    
    # arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, -1], help='-1: all folds')
    arg('--fold', type=int, default=0, choices=[0, 1, 2, 3, -1], help='-1: all folds')

    arg('--problem_type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--workers', type=int, default=2)
    arg('--test_data', type=str, default='2018', choices=['2017', '2018', 'tors'])
    
    
    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--train_crop_height', type=int, default=512) # 512 , 1024
    arg('--train_crop_width', type=int, default=640) # 640,  1280
    arg('--val_crop_height', type=int, default=512)
    arg('--val_crop_width', type=int, default=640)

    args = parser.parse_args()

    test_data = args.test_data

    if test_data == 'tors':
        file_names = get_tors_images()
    elif test_data == '2017':
        file_names = get_2017_images()
    else:
        _, file_names = get_split(args.fold)

    # model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
    #                     model_type=args.model_type, problem_type=args.problem_type) # original one


    # model = get_model(str(Path(args.model_path).joinpath('best_model.pt')),
    #                     model_type=args.model_type, problem_type=args.problem_type)

    model = get_model(str(Path(args.model_path).joinpath('detail_best_model.pt')),
                    model_type=args.model_type, problem_type=args.problem_type)

    print('num file_names = {}'.format(len(file_names)))


    # ============= Check the validation performance on valid_loader ================= #
    def make_loader(file_names, shuffle=False, transform=None, problem_type='instruments', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names))) # num train = 1639, num_val = 596

    def val_transform(p=1):
        return Compose([
            Resize(512, 640, always_apply=True, p=1),
            PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)


    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type, batch_size=args.batch_size) 
    valid_criterion = CELoss(num_classes = args.num_classes)
    valid_metrics = validation_multi(args, model, valid_criterion, valid_loader, args.num_classes) 
    print('=============================')
    print('valid_metrics', valid_metrics)
    # ================================================================================== #



    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    predict(model, file_names, args.batch_size, output_path, problem_type=args.problem_type,
            img_transform=img_transform(p=1))
