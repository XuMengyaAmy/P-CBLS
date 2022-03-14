import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

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

import sys

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

# ***************** from other .py file ******************* #
# =================================== model.py =======================================#
from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F

class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)

        x_out = f5
        return x_out

moddel_list = {'LinkNet34': LinkNet34}

# ================================= loss.py ================================== #
class CELoss(torch.nn.Module):
    def __init__(self, num_classes = None):
        super(CELoss, self).__init__()
        self.cls = torch.tensor(num_classes)
    def forward(self, outputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()   
        ce_loss = (- oh_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean() # CELoss = LogSofmax + NLLLoss 
        return ce_loss

# ================================ validation.py (Start) ============================ #
def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())

def validation_multi(args, model: nn.Module, criterion, valid_loader, num_classes, label_smoothing=None, sigma=None):
    with torch.no_grad():
        model.eval()
        losses = []
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
        for inputs, targets in valid_loader:
            inputs = cuda(inputs)
            targets = cuda(targets)
        
            outputs = model(inputs) 

            loss = criterion(outputs, targets) 

            losses.append(loss.item())
            
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)

        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.mean(list(ious.values()))
        average_dices = np.mean(list(dices.values()))

        print(
            'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss,
                                                                                   average_iou,
                                                                                   average_dices))
        metrics = {'valid_loss': valid_loss, 'iou': average_iou}
        metrics.update(ious)
        metrics.update(dices)
        return metrics


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix

def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0 
        else:
            iou = float(true_positives) / denom
        ious.append(iou)
    return ious

def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom
        dices.append(dice)
    return dices
# ================================ validation.py (End) ============================ #

# ================================ dataset.py (Start) ============================= #

from pathlib import Path
import sys
from tqdm import tqdm
import cv2
import numpy as np

# data_path = Path('data')
data_path = Path('/SurgerySegmentaion/sgmt2018/data') ################## need modify the path

def get_split(fold):

    train_path = data_path / '2018_original' / 'train'
    val_path = data_path / '2018_original' / 'val'

    train_file_names = []
    val_file_names = []

    train_file_names = list((train_path / 'images').glob('*')) 
    val_file_names = list((val_path / 'images').glob('*'))

    return train_file_names, val_file_names


import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import img_to_tensor
import albumentations as A

class RoboticsDataset(Dataset):
    def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name, self.problem_type)

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            if self.problem_type == 'binary':
                return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

binary_factor = 255
parts_factor = 85
instrument_factor = 32

def load_mask(path, problem_type):
    if problem_type == 'binary':
        mask_folder = 'binary_masks'
        factor = binary_factor
    elif problem_type == 'parts':
        mask_folder = 'parts_masks'
        factor = parts_factor
    elif problem_type == 'instruments':
        factor = instrument_factor
        mask_folder = 'instruments_masks'

    mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

    return (mask / factor).astype(np.uint8)

# ================================ dataset.py (End) ============================= #


# ====================== utils (Start) ========================= #
def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x
    
def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0

# ====================== utils (End) ========================= #

def main():
    ''' The results are reproduciable'''
    seed_everything(1234)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--fold', type=int, default=0, help='fold')
    arg('--batch_size', type=int, default=16) # 30
    arg('--lr', type=float, default=0.0001) # 0.0001, 0.00001
    arg('--workers', type=int, default=2) # 12

    arg('--train_crop_height', type=int, default=512) # 512 , 1024
    arg('--train_crop_width', type=int, default=640) # 640,  1280
    arg('--val_crop_height', type=int, default=512)
    arg('--val_crop_width', type=int, default=640)
    arg('--type', type=str, default='instruments', choices=['binary', 'parts', 'instruments'])

    arg('--model', type=str, default='LinkNet34', choices=moddel_list.keys()) # UNet:batch-size is 16. LinkNet34: batch-size is 32
    arg('--method', type=str, default='baseline', choices=['baseline', 'cbls', 'cbsvls', 'p-cbls', 'p-cbsvls'])
    arg('--model_path', type=str, default='/media/mmlab/data/mengya/SurgerySegmentaion/Segmentation_robustness/saved_best_model_segmentation/LinkNet34/Baseline/best_model.pt')

    args = parser.parse_args()
    print(args)

    print('=======================================')
    print('model            :', args.model)
    print('model_path       :', args.model_path)
    print('method           :', args.method)
    print('lr               :', args.lr)
    print('batch_size       :', args.batch_size)


    if not check_crop_size(args.train_crop_height, args.train_crop_width):
        print('Input image sizes should be divisible by 32, but train '
              'crop sizes ({train_crop_height} and {train_crop_width}) '
              'are not.'.format(train_crop_height=args.train_crop_height, train_crop_width=args.train_crop_width))
        sys.exit(0)

    if not check_crop_size(args.val_crop_height, args.val_crop_width):
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
    model_name = moddel_list[args.model]
    model = model_name(num_classes=num_classes, pretrained=True)

    model_path = args.model_path
    state = torch.load(str(model_path))
    
    state = {key.replace('module.', ''): value for key, value in state['net'].items()} # for detail_best_model
    # state = {key.replace('module.', ''): value for key, value in state.items()} # for best model
    
    model.load_state_dict(state)
    model = model.cuda()
    print('Load the saved model !!!!!!!!!!!!')

    cudnn.benchmark = True

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

    # ============= Check the validation performance on valid_loader ================= #
    valid_criterion = CELoss(num_classes = num_classes)
    valid_metrics = validation_multi(args, model, valid_criterion, valid_loader, num_classes) 
    print('=============================')
    print('valid_metrics:', valid_metrics)
    valid_iou = valid_metrics['iou']
    print('valid_iou:', valid_iou)

if __name__ == '__main__':
    main()



    # /media/mmlab/data/mengya/SurgerySegmentaion/sgmt2018/saved_model_lr_1e-4/LinkNet34/mixture/ls_0.09_factor_0.50_sig_0.90_sigfactor_0.50_ksize_3_lamda_0.80_Eall_0.40/detail_best_model.pt