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

# ========= for calculating temperature ========== #
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        self.cuda()
        ece_criterion = _ECELoss().cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Next: optimize the temperature w.r.t. NLL
        init_temp = self.temperature.clone()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        return self

class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
# ======================================================================================== #

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


def test_temperature(model, testloader, temp=1):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    logits_list = []
    labels_list = []

    with torch.no_grad():
        # for data, target in test_loader:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # print(target)
            outputs = model(inputs)
            # =========== added for temperature scaling ========== #
            # Code Reference: https://colab.research.google.com/drive/1f-cCXQbV0unhCy_bbRCYVS0wVb8Mbt4w?usp=sharing
            outputs = outputs / temp # after temperature-scaling  #################
            
            logits_list.append(outputs)
            labels_list.append(targets)
            # ===================================================== #
            test_loss += nn.CrossEntropyLoss()(outputs, targets).item()  # sum up batch loss
           
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    test_loss /= len(testloader.dataset)
    # Test_Accuracy = 100. * correct / len(testloader.dataset)
    Test_Accuracy = 100.*correct / total

    # =========== added for temperature scaling ========== #
    logits_all = torch.cat(logits_list).cuda()
    labels_all = torch.cat(labels_list).cuda()
    # ===================================================== #

    return Test_Accuracy, logits_all, labels_all
# ================================================ #


#===========================  Calculate the confidence score (Start) ===================================#
# def calculate_confidence_score(train_dataset, train_num_each, val_dataset, val_num_each, saved_model_path, temp):
def calculate_confidence_score(args, model, train_loader, temp):
    model.eval()
    logits_list = []
    labels_list = []
    true_class_wise_conf = []
    image_target_score_list_tf = []
    data_index_score_list = []

    with torch.no_grad():
        for batch_i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # =========== added for temperature scaling ========== #
            # Code Reference: https://colab.research.google.com/drive/1f-cCXQbV0unhCy_bbRCYVS0wVb8Mbt4w?usp=sharing
            output = output / temp # after temperature-scaling  #################
            logits_list.append(output)
            labels_list.append(target)
            # ===================================================== #

            # ================ calculate the confidence score  =================== #
            # Code Reference: https://github.com/mobarakol/CBLS/blob/main/CBLS.ipynb
            outputs_softmax = F.softmax(output, dim=1)
            for idx in range(len(target)):
                true_class_wise_conf.append(outputs_softmax[idx,target[idx].item()].cpu().numpy())


            image_target_score_list_tf += [{'image': image, 'target': target, 'score': score} for image, target, score in 
                                zip([data[i,::].cpu() for i in range(len(target))], [target[i].cpu().numpy().reshape(1)[0] for i in range(len(target))],
                                [outputs_softmax[idx,target[idx].item()].cpu().numpy().reshape(1)[0] for idx in range(len(target))])] 
            
            index_list = [n for n in range(batch_i*args.train_batch, batch_i*args.train_batch+len(target))]

            data_index_score_list += [{'index': index, 'score': score.item()} for index, score in zip(index_list, 
            [outputs_softmax[idx, target[idx].item()].cpu().numpy().reshape(1)[0] for idx in range(len(target))])]
    
    return true_class_wise_conf, image_target_score_list_tf, data_index_score_list
#===========================  Calculate the confidence score (End) ===================================#

import json
def write_list_to_json(list, json_file_name):
    with open(json_file_name, 'w') as  f:
        json.dump(list, f)

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
    parser.add_argument('--model', default='densenet121', help='[resnet50, densenet121]')
    parser.add_argument('--mode', default='CBLS', help='[CE, LS, CBLS, OLS, DL, SCE]')
    parser.add_argument('--use_imb', default=False, action='store_true', help='imbalance or not')
    parser.add_argument('--ls_factor_max', default=0.13, type=float, help='smoothing factor')
    parser.add_argument('--use_cbls', default=False, action='store_true', help='cbls or not')
    parser.add_argument('--no_pretrain', default=True, action='store_false', help='pretrian or not')
    parser.add_argument('--cbls_decay', default=0.9, type=float)
    parser.add_argument('--cbls_start', default=5, type=int)
    parser.add_argument('--cbls_epoch', default=5, type=int)
    
    parser.add_argument('--save_model_path', default=None, type=str) # '/home/ren2/data2/mengya/mengya_code/CBLS/CBLS_REBUTTAL/ckpt_resnet50/best_model_CE.pth.tar'
    parser.add_argument('--temp', default=None, type=float) # 1.45 for resnet50, 1.44 for densenet121
    parser.add_argument('--output_file', default=None, type=str) # 'data/tinyimagenet_resnet50_data_index_score_list.json'
    
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
        # # Display first 20 entries of resulting dictionary
        # {k: class_to_name_dict[k] for k in list(class_to_name_dict)[:20]}

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

        # train_dataset = datasets.ImageFolder(os.path.join('data/tiny-imagenet-200', 'train'),
        #     transform=transform_train,)
        # test_dataset = datasets.ImageFolder(os.path.join( 'data/tiny-imagenet-200', 'val'),
        #     transform=transform_test,)

        train_dataset = datasets.ImageFolder(os.path.join(TRAIN_DIR),
            transform=transform_train,)
        test_dataset = datasets.ImageFolder(os.path.join(val_img_dir),
            transform=transform_test,)

    train_sampler = None
    # shuffle= train_sampler is None
    # print('shuffle', shuffle) # shuffle True

    # Super important: shuffle=True for normal training. but when calculate the confidence score for sample in order, 
    # we need to set  shuffle=False

    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
    #     num_workers=2, pin_memory=True, sampler=train_sampler)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=False,
        num_workers=2, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch, shuffle=False, num_workers=2, pin_memory=True)
    print('sample size- Train:%d, Validation:%d',len(train_dataset), len(test_dataset))

    if args.mode == 'CE':
        args.ls_factor = 0.0
        criterion = CELossWithLS(classes=args.num_classes, ls_factor=args.ls_factor).to(device)
    elif args.mode == 'LS' or args.mode == 'CBLS':
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

    # Test + Temperature
    save_model_path = args.save_model_path
    print('save_model_path:', save_model_path)
    # load the model
    # checkpoint = torch.load(save_model_path+'/best_model.pt')
    checkpoint = torch.load(save_model_path)
    # model.load_state_dict(checkpoint['net'])
    model.load_state_dict(checkpoint)
    
    true_class_wise_conf, image_target_score_list, data_index_score_list = calculate_confidence_score(args, model, trainloader, temp=args.temp)

    data_index_score_list.sort(key=lambda k: (k.get('score', 0)), reverse=True) # descending order
    write_list_to_json(data_index_score_list, args.output_file)
    
if __name__ == '__main__':
    main()