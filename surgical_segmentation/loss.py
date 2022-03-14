import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import math
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


# ********************  SVLS ************************************** #
def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp( 
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = 1 if ksize==3 else 2 if ksize == 5 else 0
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()
# ******************************************************************** #


# ********** get the weights(e.g. train or not train indicator) for image in pixel-wise ************#
def get_curriculum_weights_per_px(classes, teacher_out, labels, mu, opt_t):
    out_confidence = F.softmax(teacher_out / opt_t, dim=1)
    labels_oh = F.one_hot(labels.to(torch.int64), num_classes=classes).contiguous().permute(0, 3, 1, 2)
    true_prob_all = out_confidence * labels_oh
    true_prob, _ = true_prob_all.max(dim=1, keepdim=True)
    weights = torch.where(true_prob >= mu, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
    return weights
# **************************************************************************************************#


# ******************************** Loss Zoo ****************************** #
class CELoss(torch.nn.Module):
    def __init__(self, num_classes = None):
        super(CELoss, self).__init__()
        self.cls = torch.tensor(num_classes)
    def forward(self, outputs, labels):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()   
        ce_loss = (- oh_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean() # CELoss = LogSofmax + NLLLoss 
        return ce_loss

class CELossWithLS(torch.nn.Module):
    def __init__(self, num_classes = None):
        super(CELossWithLS, self).__init__()
        self.cls = torch.tensor(num_classes)
    def forward(self, outputs, labels, label_smoothing):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()  
        smooth_oh_labels = oh_labels * (1.0 - label_smoothing) + label_smoothing / self.cls # get the smoothened target
        cels_loss = (- smooth_oh_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        return cels_loss

class CELossWithSVLS(torch.nn.Module):
    def __init__(self, num_classes=None, ksize=3):
        super(CELossWithSVLS, self).__init__()
        self.cls = torch.tensor(num_classes)
        self.ksize= ksize
    def forward(self, outputs, labels, sigma):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()    
        svls_layer = get_svls_filter_2d(ksize=self.ksize, sigma=sigma, channels=self.cls).cuda()
        svls_labels = svls_layer(oh_labels)    
        svls_loss = (- svls_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()   
        return svls_loss

class CELossWithLS_SVLS(torch.nn.Module):
    def __init__(self, num_classes = None,  ksize=3):
        super(CELossWithLS_SVLS, self).__init__()
        self.cls = torch.tensor(num_classes)
        self.ksize= ksize

    def forward(self, outputs, labels, label_smoothing, sigma):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()
        smooth_oh_labels = oh_labels * (1.0 - label_smoothing) + label_smoothing / self.cls # get the smoothened target
        svls_layer = get_svls_filter_2d(ksize=self.ksize, sigma=sigma, channels=self.cls).cuda()        
        
        ls_svls_labels = svls_layer(smooth_oh_labels)
        cels_svls_loss = (- ls_svls_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        return cels_svls_loss


class curriculum_CELossWithLS(torch.nn.Module):
    def __init__(self, num_classes = None):
        super(curriculum_CELossWithLS, self).__init__()
        self.cls = torch.tensor(num_classes)
    def forward(self, outputs, teacher_out, labels, label_smoothing, mu, opt_t):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()  
        smooth_oh_labels = oh_labels * (1.0 - label_smoothing) + label_smoothing / self.cls # get the smoothened target
        cels_loss_per_px = (- smooth_oh_labels * F.log_softmax(outputs, dim=1)).sum(dim=1) # Super important: remove the .mean()

        weights = get_curriculum_weights_per_px(self.cls, teacher_out, labels, mu, opt_t)
        curriculum_cels_loss_per_px = weights * cels_loss_per_px
        
        curriculum_cels_loss = curriculum_cels_loss_per_px.mean() # the final loss is the mean
        return curriculum_cels_loss

class curriculum_CELossWithSVLS(torch.nn.Module):
    def __init__(self, num_classes = None, ksize=3):
        super(curriculum_CELossWithSVLS, self).__init__()
        self.cls = torch.tensor(num_classes)
        self.ksize= ksize
    def forward(self, outputs, teacher_out, labels, sigma, mu, opt_t):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()    
        svls_layer = get_svls_filter_2d(ksize=self.ksize, sigma=sigma, channels=self.cls).cuda()
        svls_labels = svls_layer(oh_labels)    
        svls_loss_per_px = (- svls_labels * F.log_softmax(outputs, dim=1)).sum(dim=1) # Super important: remove the .mean()  

        weights = get_curriculum_weights_per_px(self.cls, teacher_out, labels, mu, opt_t)
        curriculum_svls_loss_per_px = weights * svls_loss_per_px

        curriculum_svls_loss = curriculum_svls_loss_per_px.mean() 
        return curriculum_svls_loss


class curriculum_CELossWithLS_SVLS(torch.nn.Module):
    def __init__(self, num_classes = None, ksize=3):
        super(curriculum_CELossWithLS_SVLS, self).__init__()
        self.cls = torch.tensor(num_classes)
        self.ksize= ksize
    def forward(self, outputs, teacher_out, labels, label_smoothing, sigma, mu, opt_t):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()  
        smooth_oh_labels = oh_labels * (1.0 - label_smoothing) + label_smoothing / self.cls # get the smoothened target

        svls_layer = get_svls_filter_2d(ksize=self.ksize, sigma=sigma, channels=self.cls).cuda()
        ls_svls_labels = svls_layer(smooth_oh_labels) # Main difference: svls_layer is applied to smooth_oh_labels, rather than oh_labels.
        
        ls_svls_loss_per_px = (- ls_svls_labels * F.log_softmax(outputs, dim=1)).sum(dim=1) # Super important: remove the .mean()

        weights = get_curriculum_weights_per_px(self.cls, teacher_out, labels, mu, opt_t)
        curriculum_ls_svls_loss_per_px = weights * ls_svls_loss_per_px
        
        curriculum_ls_svls_loss = curriculum_ls_svls_loss_per_px.mean() # the final loss is the mean
        return curriculum_ls_svls_loss


# # return modified label
class CELossWithLS_Label(torch.nn.Module):
    def __init__(self, num_classes = None):
        super(CELossWithLS_Label, self).__init__()
        self.cls = torch.tensor(num_classes)
    def forward(self, outputs, labels, label_smoothing):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()  
        smooth_oh_labels = oh_labels * (1.0 - label_smoothing) + label_smoothing / self.cls # get the smoothened target
        cels_loss = (- smooth_oh_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        return cels_loss, smooth_oh_labels


class CELossWithSVLS_Label(torch.nn.Module):
    def __init__(self, num_classes=None, ksize=3):
        super(CELossWithSVLS_Label, self).__init__()
        self.cls = torch.tensor(num_classes)
        self.ksize= ksize
    def forward(self, outputs, labels, sigma):
        oh_labels = F.one_hot(labels.to(torch.int64), num_classes = self.cls).contiguous().permute(0,3,1,2).float()    
        svls_layer = get_svls_filter_2d(ksize=self.ksize, sigma=sigma, channels=self.cls)#.cuda()
        svls_labels = svls_layer(oh_labels)    
        svls_loss = (- svls_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()   
        return svls_loss, svls_labels
