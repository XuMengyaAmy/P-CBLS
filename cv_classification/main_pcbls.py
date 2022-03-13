import os
import argparse
import copy
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models

##################
from ImageNet_dataset_pcbls.ImageNet_dataset_pcbls import ImageFolder_my
#################

from ols import OnlineLabelSmoothing
from disturblabel import DisturbLabel, SCELoss
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def seed_everything(seed=42):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

import torch
import torch.nn.functional as F
import numpy as np

class CELossWithLS(torch.nn.Module):
    def __init__(self, classes=None, ls_factor=0.1, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.ls_factor = ls_factor
        self.complement = 1.0 - self.ls_factor
        self.cls = classes
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.ignore_index = ignore_index
        #print('LS factor:',ls_factor)

    def forward(self, logits, target):
        with torch.no_grad():
            oh_labels = F.one_hot(target.to(torch.int64), num_classes = self.cls).contiguous()
            smoothen_ohlabel = oh_labels * self.complement + self.ls_factor / self.cls
        logs = self.log_softmax(logits[target!=self.ignore_index])
        return -torch.sum(logs * smoothen_ohlabel[target!=self.ignore_index], dim=1).mean()


def train(model, trainloader, criterion, optimizer, args=None):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if args.mode == 'DL':
            targets = args.disturb(targets).to(device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.*correct / total

def main():
    seed_everything()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--ls_factor', default=0.1, type=float, help='smoothing factor')
    parser.add_argument('--ckpt_dir', default='ckpt', help='checkpoint dir')
    parser.add_argument('--dataset', default='tinyimagenet', help='cifar10, cifar100, tinyimagenet')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=200) # '--num_epochs', type=int, default=100
    parser.add_argument('--train-batch', default=1024, type=int)
    parser.add_argument('--valid-batch', default=2048, type=int)
    parser.add_argument('--model', default='resnet50', help='[resnet50, densenet121]')
    parser.add_argument('--mode', default='PCBLS', help='[CE, LS, CBLS, OLS, DL, SCE, PCBLS]')
    parser.add_argument('--use_imb', default=False, action='store_true', help='imbalance or not')
    parser.add_argument('--ls_factor_max', default=0.13, type=float, help='smoothing factor')
    parser.add_argument('--use_cbls', default=False, action='store_true', help='cbls or not')
    parser.add_argument('--no_pretrain', default=True, action='store_false', help='pretrian or not')
    parser.add_argument('--cbls_decay', default=0.9, type=float)
    parser.add_argument('--cbls_start', default=5, type=int)
    parser.add_argument('--cbls_epoch', default=5, type=int)

    # PCBLS
    # parser.add_argument('--sorted_train', type=str, default='True')
    parser.add_argument('--initial_sample', type=float, default=0.6)
    parser.add_argument('--epoch_ratio', type=float, default=0.4)
    parser.add_argument('--epoch_pace', type=int, default=1)
    parser.add_argument('--json_path', type=str)
    

    args = parser.parse_args()

    print('=============================')
    print('use_cbls    ', args.use_cbls)
    print('=============================')

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_val = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    if args.dataset == 'cifar100':
        args.num_classes = 100
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    
    elif args.dataset == 'tinyimagenet':
        # =========================== Organize Validation Folder (one-loop) ======================================== #
        # Unlike training folder where images are already arranged in sub folders based 
        # on their labels, images in validation folder are all inside a single folder. 
        # Validation folder comes with images folder and val_annotations txt file. 
        # The val_annotation txt file comprises 6 tab separated columns of filename, 
        # class label, x and y coordinates, height, and width of bounding boxes
        DATA_DIR = 'data/tiny-imagenet-200'
        TRAIN_DIR = os.path.join(DATA_DIR, 'train')
        VALID_DIR = os.path.join(DATA_DIR, 'val')

        val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt', 
                            sep='\t', 
                            header=None, 
                            names=['File', 'Class', 'X', 'Y', 'H', 'W'])

        val_data.head()

        # Create separate validation subfolders for the validation images based on
        # their labels indicated in the val_annotations txt file
        val_img_dir = os.path.join(VALID_DIR, 'images')

        # Open and read val annotations text file
        fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
        data = fp.readlines()

        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        # # Display first 10 entries of resulting val_img_dict dictionary
        # {k: val_img_dict[k] for k in list(val_img_dict)[:10]}

        # Create subfolders (if not present) for validation images based on label ,
        # and move images into the respective folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

        # Save class names (for corresponding labels) as dict from words.txt file
        class_to_name_dict = dict()
        fp = open(os.path.join(DATA_DIR, 'words.txt'), 'r')
        data = fp.readlines()
        for line in data:
            words = line.strip('\n').split('\t')
            class_to_name_dict[words[0]] = words[1].split(',')[0]
        fp.close()

        print('Origanzing of valid folder  is done !!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # =================================================================================================== #
        args.num_classes = 200
        args.train_batch = 1224
        args.valid_batch = 2048
        # ============================================== #
        # UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
        transform_train = transforms.Compose([transforms.Resize(64), transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406,), (0.229, 0.224, 0.225,))]
        )
        transform_test = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406,), (0.229, 0.224, 0.225,))]
        )
        # ============================================== #

        # train_dataset = datasets.ImageFolder(os.path.join(TRAIN_DIR),
        #     transform=transform_train,)

        test_dataset = datasets.ImageFolder(os.path.join(val_img_dir),
            transform=transform_test,)

    # train_sampler = None
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
    #     num_workers=2, pin_memory=True, sampler=train_sampler)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch, shuffle=False,num_workers=2, pin_memory=True)
    # print('sample size- Train:%d, Validation:%d',len(train_dataset), len(test_dataset))
    print('sample size- Validation:', len(test_dataset))

    if args.mode == 'CE':
        args.ls_factor = 0.0
        criterion = CELossWithLS(classes=args.num_classes, ls_factor=args.ls_factor).to(device)
    elif args.mode == 'LS' or args.mode == 'CBLS' or args.mode == 'PCBLS':
        criterion = CELossWithLS(classes=args.num_classes, ls_factor=args.ls_factor).to(device)
    elif args.mode == 'OLS':
        criterion = OnlineLabelSmoothing(alpha=0.5, n_classes=args.num_classes, smoothing=0.1).to(device)
    elif args.mode == 'DL':
        args.disturb = DisturbLabel(alpha=20, C=args.num_classes) # alpha=10, 20, 40 in the original paper
        args.ls_factor = 0.0
        criterion = CELossWithLS(classes=args.num_classes, ls_factor=args.ls_factor).to(device)
    elif args.mode == 'SCE':
        criterion = SCELoss(num_classes=args.num_classes, alpha=6.0)
    else:
        raise NotImplementedError

    print('args.no_pretrain:', args.no_pretrain)
    if args.model == 'resnet50':
        print('resnet50 is preparing............')
        model = models.resnet50(pretrained=args.no_pretrain).to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model == 'densenet121':
        print('densenet121 is preparing............')
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)

    # =============== multiple-gpu training ================= #
    num_gpu = torch.cuda.device_count()
    print('=====================================')
    print('The total available GPU_number:', num_gpu)
    print('=====================================')
    if num_gpu > 1:  # has more than 1 gpu
        device_ids = np.arange(num_gpu).tolist()
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # ======================================================== #
    
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    best_epoch, best_acc = 0, 0
    init_ls = args.ls_factor
    
    # Train
    # ================== sample_pace: introduce sample_pace (float number) samples ============================== #
    # To help you understand: sample_pace = [ (1.0-0.6)* Toal_samples / ((0.4*50)/5) ] / Toal_samples 
    # sampler_pace can be calculated from other parameters
    sample_pace = float((1.0 - args.initial_sample) /
                        ((args.epoch_ratio * args.num_epochs)/args.epoch_pace))

    for epoch in range(args.num_epochs):
        if epoch != 0 and epoch < 100 and epoch % 30 == 0:
            for param in optimizer.param_groups:
                param['lr'] = param['lr'] / 10 
        if args.use_cbls:
            # args.ls_factor *= args.cbls_decay
            # ================================= #
            if epoch > 0:
                args.ls_factor *= args.cbls_decay
            # ================================== #
            args.ls_factor = max(args.ls_factor, 0)
            criterion = CELossWithLS(classes=args.num_classes, ls_factor=args.ls_factor).to(device)
        
        # ==================== PCBLS ========================== #
        if args.mode == 'PCBLS':
            num = int (epoch // args.epoch_pace) # 整除, i is the current epoch
            easy_sample_ratio = args.initial_sample + sample_pace*num
            if easy_sample_ratio > 1.0:
                easy_sample_ratio = 1.0
            
            train_dataset = ImageFolder_my(os.path.join(TRAIN_DIR), transform=transform_train, ratio=easy_sample_ratio, json_path=args.json_path)
            print('Epoch:', epoch, 'easy_sample_ratio', easy_sample_ratio, 'sample size- Train:', len(train_dataset))
            train_sampler = None
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
            num_workers=2, pin_memory=True, sampler=train_sampler)
        # ===================================================== #

        train(model, trainloader, criterion, optimizer, args=args)

        accuracy = test(model, testloader)
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            if args.mode == 'CBLS':
                torch.save(best_model.state_dict(), '{}/best_model_{}_init_ls{}_decay{}.pth.tar'.format(
                    args.ckpt_dir, args.mode, init_ls, args.cbls_decay))

            elif args.mode == 'PCBLS':
                torch.save(best_model.state_dict(), '{}/best_model_{}_{}_init_ls{}_decay{}_initial_sample{}_epoch_ratio{}.pth.tar'.format(
                    args.ckpt_dir, args.model, args.mode, init_ls, args.cbls_decay, args.initial_sample, args.epoch_ratio))
            else:
                torch.save(best_model.state_dict(), '{}/best_model_{}.pth.tar'.format(
                    args.ckpt_dir, args.mode))
        print('mode:{}:{} decay:{} epoch: {}  acc: {:.4f}  best epoch: {}  best acc: {:.4f}, ls:{:.2f}, lr:{:.4f}'.format(
                args.mode, init_ls, args.cbls_decay, epoch, accuracy, best_epoch, best_acc, args.ls_factor, optimizer.param_groups[0]['lr']))
        
        if args.mode == 'OLS':
            criterion.next_epoch()

if __name__ == '__main__':
    main()
