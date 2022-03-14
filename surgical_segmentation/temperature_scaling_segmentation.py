# ....................... Temperature Scaling in Segmentation (Start) ....................#

import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

import argparse
import json
from pathlib import Path

from loss import CELoss, CELossWithLS, CELossWithSVLS, curriculum_CELossWithLS, curriculum_CELossWithSVLS
from models import UNet11, LinkNet34, UNet, UNet16, AlbuNet, DeepLabv3_plus
from validation import validation_multi

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn



from dataset import RoboticsDataset
import utils
import sys
from prepare_data import data_path
from prepare_train_val import get_split, get_split_resized

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

moddel_list = {'UNet11': UNet11,
               'UNet16': UNet16,
               'UNet': UNet,
               'AlbuNet': AlbuNet,
               'LinkNet34': LinkNet34,
               'DeepLabv3_plus': DeepLabv3_plus}

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# *** Seed everything *** #
import torch
import random
import os

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
# ************************ #

def get_dice(labels, outputs):
    dice_arr_per_label_type = []
    for j in range(labels.size()[0]):
        label = labels.cpu().data[j].squeeze()
        label = label.squeeze().cpu().numpy()
        pred_ = outputs.cpu().data[j]

        pred = pred_.squeeze()
        out2 = pred_.data.max(0)[1].squeeze_(1)
        # print(outputs.shape, pred_.shape, out2.shape, out2.unique())
        for label_type in range(1, pred.shape[0], 1):
            dice_arr_per_label_type.append(dice_coeff(label == label_type, out2 == label_type))

    return dice_arr_per_label_type

def plot_ece_frequency(Bm_avg, title=None, n_bins=10):
    plt.figure(1)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    indx = np.arange(0, 1.1, 1/n_bins)
    plt.xticks(indx)
    plt.title(title)
    plt.bar(indx[:-1], Bm_avg/Bm_avg.sum(), width=0.08, align='edge')
    if not os.path.exists('ece'):
        os.makedirs('ece')
    plt.savefig('ece/ece_frequency_{:.3f}.png'.format(title),dpi=300)
    plt.clf()

def reliability_diagram(conf_avg, acc_avg, title=None, leg_idx=0, n_bins=10):
    plt.figure(2)
    #plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot([conf_avg[acc_avg>0][0], 1], [conf_avg[acc_avg>0][0], 1], linestyle='--')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    #plt.xticks(np.arange(0, 1.1, 1/n_bins))
    #plt.title(title)
    plt.plot(conf_avg[acc_avg>0],acc_avg[acc_avg>0], marker='.', label = title)
    plt.legend()
    if not os.path.exists('ece'):
        os.makedirs('ece')
    plt.savefig('ece/ece_reliability_{:.3f}.png'.format(title),dpi=300)

def get_ece(preds, targets, n_bins=10, ignore_bg = False):
    #ignore_bg = False to ignore bckground class from ece calculation
    bg_cls = 0 if ignore_bg else -1
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    confidences, predictions = confidences[targets>bg_cls], predictions[targets>bg_cls]
    accuracies = (predictions == targets[targets>bg_cls]) 
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        Bm[bin_idx] = bin_size
        if bin_size > 0:  
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        bin_idx += 1
        
    ece_all = Bm * np.abs((acc - conf))/ Bm.sum()
    ece = ece_all.sum() 
    return ece, acc, conf, Bm

def dice_coeff(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    intersection = np.sum(np.logical_and(y_true, y_pred))
    smooth = 0.0001
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def get_temperature_scaling(model, dataloaders, args):
    model.eval()
    loss_ce = torch.nn.CrossEntropyLoss()
    best_ece = np.inf
    opt_T = 1
    const_loss = 0
    start = args.init_t
    inc = args.inc
    end = args.end_t
    T_list = np.arange(start, end, inc)
    with torch.no_grad():
        for T in T_list:
            ece, acc_all, conf_all, bm_all = [], [], [], []
            losses = []
            dice_arr = []
            for idx, (inputs, labels) in enumerate(dataloaders):
                inputs, labels = inputs.cuda(), labels.cuda()
                
                logits = model(inputs)

                dice_arr.append(np.mean(get_dice(labels, logits)))
                loss = loss_ce(logits, labels.long())
                losses.append(loss.item())
                
                pred_conf = F.softmax(logits/T, dim=1).detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                ece_per_batch, acc, conf, bm = get_ece(pred_conf, labels, ignore_bg = False)
                ece.append(ece_per_batch)
                acc_all.append(acc)
                conf_all.append(conf)
                bm_all.append(bm)
                
            ece_avg = np.stack(ece).mean(0)
            acc_avg = np.stack(acc_all).mean(0)
            conf_avg = np.stack(conf_all).mean(0)
            bm_avg = np.stack(bm_all).mean(0)
            losses_avg = round(np.mean(losses),4)

            if T == start:
                best_ece = ece_avg
                opt_T = T
                const_loss = losses_avg
                reliability_diagram(conf_avg, acc_avg, title=T)
                plot_ece_frequency(bm_avg, title=T)
                
            if const_loss != losses_avg:
                break
            if ece_avg < best_ece and const_loss == losses_avg and T is not 1:
                best_ece = ece_avg
                opt_T = T
                reliability_diagram(conf_avg, acc_avg, title=T)
                plot_ece_frequency(bm_avg, title=T)
            print('current loss:%.4f, ece:%.4f, T:%.2f, best ece:%.4f, opt_T:%.2f, const_loss:%.4f, dice:%.4f'
                    %(losses_avg, ece_avg, T, best_ece, opt_T, const_loss, np.mean(dice_arr)))
            
    return opt_T

# ....................... Temperature Scaling in Segmentation (End) ....................#
def main():
    ''' The results are reproduciable'''
    seed_everything(1234)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int, default=0, help='fold')
    arg('--batch_size', type=int, default=32) # 16
    arg('--n_epochs', type=int, default=100, help='the number of total epochs')
    arg('--lr', type=float, default=0.0001) 
    arg('--workers', type=int, default=2) 

    ########################################################
    # For LinkNet34 model
    arg('--train_crop_height', type=int, default=512) # 1024, 512, 204,
    arg('--train_crop_width', type=int, default=640) # 1280, 640,  256
    arg('--val_crop_height', type=int, default=512) # # 1024, 512, 204,
    arg('--val_crop_width', type=int, default=640) # 1024, 512, 204,

    # # For DeepLabv3_plus model，we hope it's torch.Size([16, 3, 204, 256])
    # arg('--train_crop_height', type=int, default=204) # 1024, 512, 204,
    # arg('--train_crop_width', type=int, default=256) # 1280, 640,  256

    # arg('--val_crop_height', type=int, default=204) # 1024, 512, 204,
    # arg('--val_crop_width', type=int, default=256) # 1280, 640,  256
    ########################################################


    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])

    arg('--model', type=str, default='LinkNet34', choices=moddel_list.keys()) # DeepLabv3_plus, LinkNet34
    arg('--method', type=str, default='baseline', choices=['baseline', 'cbls', 'cbsvls', 'p-cbls', 'p-cbsvls'])

    arg('--resume_last', type=str, default = 'False', help='resume the training from the last epoch')
    
    # CBLS
    arg('--label_smoothing', default=0.3, type=float)
    arg('--factor', default=0.9, type=float) # 0.95

    # ************** New added ********************* #
    # CBSVLS
    arg('--sigma', default=1.0, type=float)
    arg('--sigma_factor', default=0.5, type=float) # sigma should be around 0 when at 50 % of the total epochs. In our case, it should be epoch 50. # math.pow(0.01, 49)=1e-98 which is around 0

    # Pixel-wise paced learning
    arg('--opt_t', default=2.53, type=float)
    arg('--lamda', default=0.9, type=float)
    arg('--E_all', default=0.3, type=float)
    arg('--mu', default=0.8, type=float)
    arg('--mu_update', default=0.02, type=float)
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
    print('mu               :', args.mu)
    print('mu_update        :', args.mu_update)
    # we don't need to check the crop size for DeepLabv3_plus model
    if args.method == 'LinkNet34': # LinkNet34 model
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

    elif args.model == 'DeepLabv3_plus':
        print('11111111111111111111')
        model_name = moddel_list[args.model]
        model = model_name(n_classes=num_classes, pretrained=True)  
    else:
        model_name = moddel_list[args.model]
        model = model_name(num_classes=num_classes, pretrained=True)

    model_path = 'saved_model_lr_1e-4/LinkNet34/baseline/best_model.pt' 
    # model_path = 'saved_model_lr_1e-4/DeepLabv3_plus/baseline/detail_best_model.pt'

    print('model_path:', model_path)
    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['net'].items()} # original one
    # state = {key.replace('module.', ''): value for key, value in state.items()}
    
    model.load_state_dict(state)
    model = model.cuda()
    print('Load the saved model !!!!!!!!!!!!')

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, problem_type='binary', batch_size=1):
        return DataLoader(
            dataset=RoboticsDataset(file_names, transform=transform, problem_type=problem_type),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    if args.model == 'DeepLabv3_plus':
        train_file_names, val_file_names = get_split_resized(args.fold)
    else:
        train_file_names, val_file_names = get_split(args.fold) # please enable it for LinkNet34

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names))) # num train = 1639, num_val = 596
    
    if args.model == 'DeepLabv3_plus': # Input for DeepLabv3_plus are resized images, so we don't need the Resize here
        def val_transform(p=1):
            return Compose([
                PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
                CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
                Normalize(p=1)
            ], p=p)

    else:
        def val_transform(p=1):
            return Compose([
                Resize(512, 640, always_apply=True, p=1),
                PadIfNeeded(min_height=args.val_crop_height, min_width=args.val_crop_width, p=1),
                CenterCrop(height=args.val_crop_height, width=args.val_crop_width, p=1),
                Normalize(p=1)
            ], p=p)


    valid_loader = make_loader(val_file_names, transform=val_transform(p=1), problem_type=args.type, batch_size=args.batch_size) 
    # ============= Check the validation performance on valid_loader ================= #
    valid_criterion = CELoss(num_classes = num_classes)
    valid_metrics = validation_multi(args, model, valid_criterion, valid_loader, num_classes) 
    print('=============================')
    print('valid_metrics', valid_metrics)

    #########################################################################
    # # get the optimal T for the temperature scaling
    # args.init_t = 1
    # args.inc = 0.2 # original one is args.inc = 0.5
    # args.end_t = 5
    # opt_T = get_temperature_scaling(model, valid_loader, args) 
    
    #########################################################################

    # Fine-tune the optimal T for the temperature scaling
    opt_T = 1.00
    args.init_t = opt_T - 0.5
    args.inc = 0.1
    args.end_t = opt_T + 0.5
    opt_T = get_temperature_scaling(model, valid_loader, args) 

    # #########################################################################
    
    # # # # Fine-tune the optimal T for the temperature scaling
    # # opt_T = 1.60 # for 70 epochs

    # opt_T = 0.90
    # args.init_t = opt_T - 0.1
    # args.inc = 0.01
    # args.end_t = opt_T + 0.1
    # opt_T = get_temperature_scaling(model, valid_loader, args) # opt_T:1.60  


if __name__ == '__main__':
    main()