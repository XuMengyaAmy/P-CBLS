import argparse
import json
from pathlib import Path

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet
from loss import CELoss
from validation import validation_multi


from dataset import RoboticsDataset
import utils
import sys
from prepare_data import data_path
from prepare_train_val import get_split, get_split_order

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Normalize,
    Compose,
    PadIfNeeded,
    RandomCrop,
    CenterCrop,  
    Resize
)

import warnings
warnings.filterwarnings("ignore")

import math
import pathlib

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
# ****************** Constant Seed ****************** #
import random
import numpy as np

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(1234)

# ****************** Constant Seed ****************** #

def main():
    ''' The results are reproduciable'''
    print('Seed everything')
    seed_everything(1234)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int, default=0, help='fold')
    arg('--batch_size', type=int, default=16) 
    arg('--n_epochs', type=int, default=70, help='the number of total epochs')
    arg('--lr', type=float, default=0.0001) 
    arg('--workers', type=int, default=2) 

    arg('--train_crop_height', type=int, default=512) 
    arg('--train_crop_width', type=int, default=640) 
    arg('--val_crop_height', type=int, default=512)
    arg('--val_crop_width', type=int, default=640)

    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--num_classes', type=int, default=8) # instruments segmentation, num_classes = 8

    arg('--model', type=str, default='LinkNet34', choices=moddel_list.keys()) 
    arg('--method', type=str, default='baseline', choices=['baseline', 'cbls', 'cbsvls', 'p-cbls', 'p-cbsvls'])

    arg('--resume_last', type=str, default = 'False', help='resume the training from the last epoch')
    
    # CBLS
    arg('--label_smoothing', default=0.6, type=float)
    arg('--factor', default=0.9, type=float) # 0.95

    # CBSVLS
    arg('--sigma', default=1.0, type=float)
    arg('--sigma_factor', default=0.6, type=float) 

    # Pixel-wise paced learning
    arg('--opt_t', default=1.60, type=float)
    arg('--lamda', default=0.9, type=float)
    arg('--E_all', default=0.4, type=float)

    args = parser.parse_args()
    print(args)

    print('=======================================')
    print('model            :', args.model)
    print('method           :', args.method)
    print('n_epochs         :', args.n_epochs)
    print('lr               :', args.lr)
    print('batch_size       :', args.batch_size)
    print('============== CBLS =====================')
    print('label_smoothing  :', args.label_smoothing)
    print('factor           :', args.factor)
    print('=============== CBSVLS ====================')
    print('sigma            :', args.sigma)
    print('sigma_factor     :', args.sigma_factor)
    print('======= pixel-wsie paced learning ==========')
    print('opt_t            :', args.opt_t)
    print('lamda            :', args.lamda)
    print('E_all            :', args.E_all)

    if not utils.check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not utils.check_crop_size(args.val_crop_height, args.val_crop_width):
        print('Input image sizes should be divisible by 32, but validation '
              'crop sizes ({val_crop_height} and {val_crop_width}) '
              'are not.'.format(val_crop_height=args.val_crop_height, val_crop_width=args.val_crop_width))
        sys.exit(0)

    # num classes
    if args.type == 'parts':
        num_classes = 4
    elif args.type == 'instruments': # only train instrument segmentation
        print('instruments segmentation')
        num_classes = 8  # 7 instruments + background
    else:
        num_classes = 1
    
    # initialize model
    if args.model == 'UNet':
        model = UNet(num_classes=num_classes)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True) 


    # ****** baseline + temperature-scaling (later) *********** #
    model_path = 'saved_model_70epochs_lr_1e-4/LinkNet34/baseline/best_model.pt'
    print('model_path:', model_path)
    state = torch.load(str(model_path))
    # state = {key.replace('module.', ''): value for key, value in state['net'].items()} # original one
    state = {key.replace('module.', ''): value for key, value in state.items()}
    model.load_state_dict(state)
    print('Load the saved model !!!!!!!!!!!!')

    model = model.cuda()


    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    #train_file_names, val_file_names = get_split(args.fold) # the dataset is not ordered
    train_file_names, val_file_names = get_split_order(args.fold) # get the ordered dataset
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names))) # num train = 1639, num_val = 596
    print('last two train_file_names', train_file_names[-2], train_file_names[-1])
    print('last two val_file_names', val_file_names[-2], val_file_names[-1])

    def train_transform(p=1):
        return Compose([
            Resize(512, 640, always_apply=True, p=1),
            PadIfNeeded(min_height=args.train_crop_height, min_width=args.train_crop_width, p=1),
            RandomCrop(height=args.train_crop_height, width=args.train_crop_width, p=1),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            Resize(512, 640, always_apply=True, p=1),
            PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
            CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
            Normalize(p=1)
        ], p=p)

 
    train_loader = make_loader(train_file_names, shuffle=False, transform=train_transform(p=1), problem_type=args.type,
                               batch_size=args.batch_size) # Super important: Remember to set shuffle = False when calculating the task difficulty
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type, batch_size=args.batch_size)

    # ============= Calculate the task difficulty for train loader. Remember to set shuffle = False =================== #
    from torch.nn import functional as F
    from tqdm import tqdm
    import numpy as np

    def calculate_class_conf(args, model: nn.Module, criterion, data_loader, num_classes):
        true_class_wise_conf = []
        
        # non_0_label_in_one_sample = []
        with tqdm(desc='calculate confidence', unit='it', total=len(data_loader)) as pbar:
            with torch.no_grad():
                model.eval()
                for inputs, targets in data_loader:
                    inputs = utils.cuda(inputs)
                    targets = utils.cuda(targets)
                    
                    outputs_raw = model(inputs) 
                    # print('outputs_raw', outputs_raw)
                    # print(outputs_raw.shape) # torch.Size([32, 8, 512, 640])
                    
                    # ***************** obtain the confidence with or without temperature-scaling ************************** #
                    # # without temperature-scaling
                    # outputs_softmx = F.softmax(outputs_raw, dim=1) # torch.Size([32, 8, 512, 640]), probability within 0 to 1
                    
                    # with temperature-scaling
                    outputs_softmx = F.softmax(outputs_raw / args.opt_t, dim=1)
                    # ******************************************************************************************************* #


                    # print('outputs_softmx', outputs_softmx)  
                    # print(outputs_softmx.shape)

                    for idx in range(outputs_softmx.shape[0]): # batch_szie = 32
                        true_class_probability_sum_one_sample = 0
                    
                        output = outputs_softmx[idx]
                        # print('output for one sample', output)
                        # print('output for one sample', output.shape) # torch.Size([8, 512, 640])
                        target = targets[idx]
                        # print('target for one sample', target)
                        # print('target for one sample', target.shape) # torch.Size([512, 640])
                        # Calculate the confidence score for one image
                        
                        print('unique class in target', np.unique(np.array(target.cpu()))) #####
                        
                        # # '==================== method 1 (too slow): calculate confidence score  ==================='
                        # foreground_pixel_num = 0
                        # # # find the corresponding probability in output
                        # for i in range(target.shape[0]):
                        #     for j in range(target.shape[1]):
                        #         # label_h_w = {}
                        #         if target[i][j] != 0:
                        #             foreground_pixel_num += 1
                        #             true_class_probability_sum_one_sample += output[target[i][j], i, j].item()
                        #             # label_h_w['label'], label_h_w['h'], label_h_w['w'] = target[i,j], i, j
                        #             # non_0_label_in_one_sample.append(label_h_w)
                        
                        # if foreground_pixel_num == 0: # the image has no instruments
                        #     confidence = 0.0
                            
                        # else:
                        #     confidence = true_class_probability_sum_one_sample / foreground_pixel_num
                        # print('confidence', confidence) ##########
                        # true_class_wise_conf.append(confidence)
                        # '======================================================================'

                        # '======================== method 2 (very fast): calculate confidence score  =====================' #
                        # Code Reference: https://www.geeksforgeeks.org/numpy-nonzero-in-python/
                        target_arr = np.array(target.cpu())
                        if np.all(target_arr == 0): # no instruments class in this image
                            confidence = 0.0
                        else:
                            foreground_class = target_arr[np.nonzero(target_arr)] # arr[geek.nonzero(arr)]
                            foreground_pixel_index = np.transpose(np.nonzero(target_arr))# geek.transpose(geek.nonzero(arr))
                            # print('foreground_class', foreground_class) # <class 'numpy.ndarray'>,the shape is (53874,)
                            for i in range(foreground_class.shape[0]):
                                true_class_probability_sum_one_sample += output[foreground_class[i], foreground_pixel_index[i][0], foreground_pixel_index[i][1]].item()
                            confidence = true_class_probability_sum_one_sample / foreground_class.shape[0]
                        print('confidence', confidence)
                        true_class_wise_conf.append(confidence)
                        # '======================================================================'
                    pbar.update()
        return true_class_wise_conf
    # ========================================================================================================================= #
    
    # ============= Check the validation performance on valid_loader ================= #
    valid_criterion = CELoss(num_classes = num_classes)
    valid_metrics = validation_multi(args, model, valid_criterion, valid_loader, num_classes) # based on valid_loader
    print('=============================')
    print('valid_metrics', valid_metrics)

    true_class_wise_conf = calculate_class_conf(args, model, valid_criterion, train_loader, num_classes) # based on train_loader
    print('Train Size:', len(true_class_wise_conf))
    print(true_class_wise_conf[0:25])
    


    file_path = 'data/sample_wise_data/LinkNet34/'
    if not os.path.isdir(file_path):
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    
    filename = open(os.path.join(file_path,'ts_confidence_score.txt'),'w')
    for i in true_class_wise_conf:
        filename.write(str(i))
        filename.write('\n')
    filename.close()
    print('Done')



if __name__ == '__main__':
    main()

