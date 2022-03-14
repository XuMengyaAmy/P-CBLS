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

from models import UNet, UNet11, LinkNet34, UNet16, AlbuNet

from loss import CELoss, CELossWithLS, CELossWithSVLS, curriculum_CELossWithLS, curriculum_CELossWithSVLS
from validation import validation_multi

from dataset import RoboticsDataset
import utils
import sys
from prepare_data import data_path
from prepare_train_val import get_split, get_split_SPL

import warnings
warnings.filterwarnings("ignore")
import math

# ****************** Constant Seed ****************** #
import random
import numpy as np

def seed_everything(seed=1234):
    print('Seed everything')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything(1234)
# ****************** Constant Seed ****************** #

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

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34}

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def main():
    ''' The results are reproduciable'''
    seed_everything(1234)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int, default=0, help='fold')
    arg('--batch_size', type=int, default=16) 
    arg('--n_epochs', type=int, default=70, help='the number of total epochs')
    arg('--lr', type=float, default=0.0001) 
    arg('--workers', type=int, default=2) 

    arg('--train_crop_height', type=int, default=512) # 512 , 1024
    arg('--train_crop_width', type=int, default=640) # 640,  1280
    arg('--val_crop_height', type=int, default=512)
    arg('--val_crop_width', type=int, default=640)

    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])
    arg('--num_classes', type=int, default=8) # instruments segmentation, num_classes = 8

    arg('--model', type=str, default='LinkNet34', choices=moddel_list.keys()) # UNet:batch-size is 16. LinkNet34: batch-size is 32
    arg('--method', type=str, default='samplewise-p-cbls', choices=['baseline', 'cbls', 'cbsvls', 'p-cbls', 'p-cbsvls', 'samplewise-p-cbls', 'samplewise-p-cbsvls'])

    arg('--resume_last', type=str, default = 'False', help='resume the training from the last epoch')
    
    # CBLS
    arg('--label_smoothing', default=0.6, type=float)
    arg('--factor', default=0.9, type=float) # 0.95

    # CBSVLS
    arg('--sigma', default=1.0, type=float)
    arg('--sigma_factor', default=0.6, type=float) # sigma should be around 0 when at 50 % of the total epochs. In our case, it should be epoch 50. # math.pow(0.01, 49)=1e-98 which is around 0
    arg('--ksize', default=3, type=int)
    
    # Pixel-wise paced learning  
    arg('--opt_t', default=1.60, type=float)
    arg('--lamda', default=0.9, type=float)
    arg('--E_all', default=0.4, type=float)


    # Sample-wise paced learning
    # Linear MPL (Linear Multi-Paced Learning)
    arg('--train_file_spl', type=str, default='data/sample_wise_data/LinkNet34/sorted_image_path_ts_confidence.lst') # change for LinkNet34
    arg('--initial_sample', default=0.6, type=float)
    arg('--all_sample_epoch_ratio', default=0.4, type=float, help= 'epoch when introducing all samples / n_epochs')
    arg('--epoch_pace', default=1, type=int) # epoch_pace is fixed most of the time # 2

    args = parser.parse_args()
    print(args)

    # ================== sample_pace: introduce sample_pace (float number) samples ============================== #
    # To help you understand: sample_pace = [ (1.0-0.6)* Toal_samples / ((0.4*50)/5) ] / Toal_samples 
    # sampler_pace can be inferred from other parameters
    sample_pace = float( (1.0 - args.initial_sample) / ((args.all_sample_epoch_ratio * args.n_epochs)/args.epoch_pace) ) # 1.0 means 100% samples
    
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
    print('ksize            :', args.ksize)
    print('======= pixel-wsie paced learning ==========')
    print('opt_t            :', args.opt_t)
    print('lamda            :', args.lamda)
    print('E_all            :', args.E_all)
    print('======= Sample-wsie paced learning ==========')
    print('train_file_spl          :', args.train_file_spl)
    print('initial_sample          :', args.initial_sample)
    print('all_sample_epoch_ratio  :', args.all_sample_epoch_ratio)
    print('epoch_pace              :', args.epoch_pace)
    print('sample_pace             :', sample_pace)


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
        if args.method == 'p-cbls' or args.method == 'p-cbsvls':
            teacher_model = UNet(num_classes=num_classes)
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)  
        if args.method == 'p-cbls' or args.method == 'p-cbsvls':
            teacher_model = model_name(num_classes=num_classes, pretrained=True)

    # assign GPU device
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        print('The total available GPU_number:', num_gpu)
        if num_gpu > 0:
            device_ids = np.arange(num_gpu).tolist() 
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        
        if args.method == 'p-cbls' or args.method == 'p-cbsvls':
            # ======= pixel-wise curriculum ========#
            teacher_model = nn.DataParallel(teacher_model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    # loss function
    if args.type == 'instruments':
        if args.method == 'baseline':
            print('=================== Using CELoss ========================')
            loss = CELoss(num_classes = num_classes)
        elif args.method == 'cbls' or args.method == 'samplewise-p-cbls':
            print('=================== Using CELossWithLS ========================')
            loss = CELossWithLS(num_classes = num_classes)
        elif args.method == 'cbsvls' or args.method == 'samplewise-p-cbsvls':
            print('=================== Using CELossWithSVLS ========================')
            loss = CELossWithSVLS(num_classes = num_classes, ksize = args.ksize)
        elif args.method == 'p-cbls':
            print('=================== Using curriculum_CELossWithLS ========================')
            loss = curriculum_CELossWithLS(num_classes = num_classes)
        elif args.method == 'p-cbsvls':  
            print('=================== Using curriculum_CELossWithSVLS ========================')
            loss = curriculum_CELossWithSVLS(num_classes = num_classes, ksize = args.ksize)


    # cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)
    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names))) # num train = 1639, num_val = 596

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

    # train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1), problem_type=args.type, batch_size=args.batch_size) # original one
    
    if args.method == 'samplewise-p-cbls' or args.method == 'samplewise-p-cbsvls':
        ''' =======================  Sample-Wise Linear MPL =========================== '''
        train_loader_all = []

        import math
        max_num = math.ceil(args.n_epochs/args.epoch_pace) # length of list - train_loader_all; when args.epoch_pace=1, max_num = args.n_epochs
        print('max_num:', max_num)
        
        for i in range(max_num):
            easy_sample_ratio = args.initial_sample + sample_pace*i
            if easy_sample_ratio < 1:
                train_loader_all.append(make_loader(get_split_SPL(args.fold, args.train_file_spl, easy_ratio = easy_sample_ratio), 
                                                    shuffle=True, transform=train_transform(p=1), problem_type=args.type, batch_size=args.batch_size))
            else:
                easy_sample_ratio = 1.0
                train_loader_all.append(make_loader(get_split_SPL(args.fold, args.train_file_spl, easy_ratio = 1.0), 
                                                    shuffle=True, transform=train_transform(p=1), problem_type=args.type, batch_size=args.batch_size))
        print('train_loader_all list', len(train_loader_all))
        ''' ================================================================ '''
   
   
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type, batch_size=args.batch_size) 

    valid = validation_multi

    # ================ optimizer  ======================= #
    optimizer = Adam(model.parameters(), lr = args.lr)

    if args.method == 'baseline' or args.method == 'cbls' or args.method == 'cbsvls':
        print('******************************** without pixel-wise paced learning ****************************')
        utils.train(
            args=args,
            model=model,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=valid,
            num_classes=num_classes,
            optimizer = optimizer
        )
    elif args.method == 'p-cbls' or args.method == 'p-cbsvls':
        print('******************************** with pixel-wise paced learning *********************************')
        # ======= pixel-wise curriculum ========#
        teacher_model.load_state_dict(torch.load('saved_model_70epochs_lr_1e-4/LinkNet34/baseline/best_model.pt'))
        args.mu, args.mu_update = utils.get_threshold(teacher_model, train_loader, args)
        print("initial_mu:{:.4f}, mu_update:{:.4f}".format(args.mu, args.mu_update))
        
        # 100 epochs (due to the mistake)
        # percentage of pixels found: 0.902
        # initial_mu:0.9550, mu_update:0.0318

        # 70 epochs
        # percentage of pixels found: 0.902
        # initial_mu:0.9550, mu_update:0.0455

        utils.train_pixel_wise_curriculum(
            args=args,
            model=model,
            teacher_model=teacher_model,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=valid,
            num_classes=num_classes,
            optimizer = optimizer,
        )
    elif args.method == 'samplewise-p-cbls' or args.method == 'samplewise-p-cbsvls':
        print('******************************** with sample-wise paced learning ****************************')
        utils.train_sample_wise_curriculum(
            args=args,
            model=model,
            criterion=loss,
            train_loader_all=train_loader_all,
            valid_loader=valid_loader,
            validation=valid,
            num_classes=num_classes,
            optimizer = optimizer
        )
    

if __name__ == '__main__':
    main()
