import json
from datetime import datetime
from pathlib import Path
import numpy as np
import random
import torch
import tqdm
import os

from torch.optim import Adam

from loss import CELoss

import pathlib


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

def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def train(args, model, criterion, train_loader, valid_loader, validation, num_classes, optimizer):
    seed_everything(1234)
    valid_criterion = CELoss(num_classes = num_classes)
    
    n_epochs = args.n_epochs
    
    valid_losses = []
    
    if args.method == 'baseline':
        save_model_path = 'saved_model_lr_1e-4/%s/%s' % (args.model, args.method)
    elif args.method == 'cbls' or args.method =='linear_cbls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f' % (args.model, args.method, args.label_smoothing, args.factor)
    elif args.method == 'cbsvls' or args.method == 'linear_cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.3f_ksize_%d' % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize)
    elif args.method == 'cbls_cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_sig_%.2f_sigfactor_%.2f_ksize_%d' % (args.model, args.method, args.label_smoothing, args.factor, args.sigma, args.sigma_factor, args.ksize)

    if not os.path.isdir(save_model_path):
        pathlib.Path(save_model_path).mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_epoch = 0
    start_epoch = 0 # if training from scratch (not resume training)

    if args.resume_last == 'True':
        #------ last model path -------#
        path_checkpoint = 'LinkNet34/baseline/model_epoch_40.pt'
        if os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)

            model.load_state_dict(checkpoint['net'])           # load the model's parameters
            optimizer.load_state_dict(checkpoint['optimizer']) # load the optimizer's parameters
            start_epoch = checkpoint['epoch'] + 1              # set the start epoch, rememr to add 1
            
            # ========================== Validation ========================== #
            # confirm it is last epoch based on the metric
            valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
            
            # the best_iou in previous experiments is xxx (need to manually update)
            #--- baseline (LinkNer34) ---#
            best_iou =  0.43643259266831347
            best_epoch = 39
            # =================================================================== #
            print('Resuming from epoch %d' % checkpoint['epoch'])

    for epoch in range(start_epoch, n_epochs):
        # update the parameters every epoch
        # ======================= CBLS (decrease) ================== #
        if args.method == 'cbls':
            if epoch == 0:
                label_smoothing = args.label_smoothing
            elif label_smoothing > 0.0:
                label_smoothing *= args.factor # decrease exponentially
            else:
                label_smoothing = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' label_smoothing:', label_smoothing)
        # ============================================================ #


        # ===================== Linear CBLS (decrease) ========================= #
        elif args.method == 'linear_cbls':
            if epoch == 0:
                label_smoothing = args.label_smoothing
            elif label_smoothing > 0.0:
                label_smoothing -= args.factor  
                # decrease linearly (It cannot be ruled out that it becomes negative after the subtraction operation)
                # Epoch 7  label_smoothing: -0.09999999999999998
                if label_smoothing < 0.0:
                    label_smoothing = 0.0
            else:
                label_smoothing = 0.0 # 0.0 is minimum

            # # use the round to solve the problem of the float type in python; round(1.234, 2) = 1.23, remove 0 automatically, 进位采用四舍五入(round)
            label_smoothing = round(label_smoothing, 2) # the validation results seems similar when with and without the round operation

            print('Epoch', epoch, ' linear_cbls:   label_smoothing:', label_smoothing)            
        # ============================================================ #

        # ======================= CBSVLS (decrease) ================== #
        elif args.method == 'cbsvls':
            if epoch == 0:
                sigma = args.sigma
            elif sigma > 0.0:
                sigma *= args.sigma_factor
            else:
                sigma = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' sigma:', sigma)
        # ============================================================ #


        # ===================== Linear CBLS(SVLS) (decrease) ========================= #
        elif args.method == 'linear_cbsvls':
            if epoch == 0:
                sigma = args.sigma
            elif sigma > 0.0:
                sigma -= args.sigma_factor
                # decrease linearly (It cannot be ruled out that it becomes negative after the subtraction operation)
                # Epoch 7  label_smoothing: -0.09999999999999998
                if sigma < 0.0:
                    sigma = 0.0
            else:
                sigma = 0.0 # 0.0 is minimum

            # # use the round to solve the problem of the float type in python; round(1.234, 2) = 1.23, remove 0 automatically, 进位采用四舍五入(round)
            sigma = round(sigma, 2) # the validation results seems similar when with and without the round operation

            print('Epoch', epoch, ' linear_cbsvls:   sigma:', sigma)            
        # ============================================================ #


        # ========================= cbls_cbsvls = CBLS (ULS+SVLS) ================================= #
        # "SVLS on top of ULS": decrease both ULS and SVLS factor during whole training, then we need to combine their loss.
        elif args.method == 'cbls_cbsvls':
            if epoch == 0:
                label_smoothing = args.label_smoothing
                sigma = args.sigma

            elif label_smoothing > 0.0 and sigma > 0.0:
                label_smoothing *= args.factor
                sigma *= args.sigma_factor

                if label_smoothing < 0.0:
                    label_smoothing = 0.0
                if sigma < 0.0:
                    sigma = 0.0
            else:
                label_smoothing = 0.0
                sigma = 0.0
            print('Epoch', epoch, ' label_smoothing:', label_smoothing, ' sigma:', sigma)
        # ==========================================================================#

        model.train()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        # tq.set_description('Epoch {}'.format(epoch))
        
        losses = []
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            # print('inputs shape:', inputs.shape)
            outputs = model(inputs)

            if args.method == 'baseline':
                loss = criterion(outputs, targets) ###############################################
            elif args.method == 'cbls' or args.method =='linear_cbls':
                loss = criterion(outputs, targets, label_smoothing)
            elif args.method == 'cbsvls' or args.method == 'linear_cbsvls':
                loss = criterion(outputs, targets, sigma)
            elif args.method == 'cbls_cbsvls':
                loss = criterion(outputs, targets, label_smoothing, sigma)

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()
            
            tq.update(batch_size)
            losses.append(loss.item())

        tq.close()

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']

        if args.method == 'baseline':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss
                }
        elif args.method == 'cbls' or args.method =='linear_cbls':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "label_smoothing": label_smoothing,
                    "factor": args.factor 
                }
        elif args.method == 'cbsvls' or args.method == 'linear_cbsvls':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "sigma": sigma,
                    "sigma_factor": args.sigma_factor
                }

        elif args.method == 'cbls_cbsvls':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "label_smoothing": label_smoothing,
                    "factor": args.factor,
                    "sigma": sigma,
                    "sigma_factor": args.sigma_factor
                } # record the smoothing and sigma value of current epoch

        if valid_iou > best_iou:
            best_iou = valid_iou
            best_epoch = epoch
            print('===================saving the best model!========================')
            # torch.save(model.state_dict(), os.path.join(save_model_path, "best_model.pt"))

            # saved details
            torch.save(checkpoint, os.path.join(save_model_path, "detail_best_model.pt"))
        
        if (epoch+1) % 20 == 0:
            # path_to_save = os.path.join(save_model_path, "model_epoch_%02d.pt" % epoch)
            # torch.save(model.state_dict(), path_to_save)

            # saved details
            path_to_save_detail = os.path.join(save_model_path, "detail_model_epoch_%02d.pt" % epoch)
            torch.save(checkpoint, path_to_save_detail)

        elif epoch == n_epochs-1:
            path_to_save_detail = os.path.join(save_model_path, "last_model_epoch_%02d.pt" % epoch)
            torch.save(checkpoint, path_to_save_detail)
        # ======================================================= # 
        print('best epoch unitl now:', best_epoch, ' best iou unitl now:', best_iou)

    print('best epoch:', best_epoch, ' best iou:', best_iou)


from itertools import accumulate
import torch.nn.functional as F
def get_threshold(teacher_model, dataloaders, args):
    teacher_model.eval()
    bin_size = 1000
    conf_values = np.arange(1 / bin_size, 1 + 1 / bin_size, 1 / bin_size)
    hist_prob = torch.zeros(len(conf_values))
    mu, mu_update = 0, 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders):
            inputs, labels = inputs.cuda(), labels.cuda()
            teacher_out = teacher_model(inputs)
            out_confidence = F.softmax(teacher_out / args.opt_t, dim=1)
            labels_oh = F.one_hot(labels.to(torch.int64), num_classes=args.num_classes).contiguous().permute(0, 3, 1, 2)
            true_prob_all = out_confidence * labels_oh
            true_prob, _ = true_prob_all.cpu().max(dim=1, keepdim=True)
            hist_prob += torch.histc(true_prob, bins=bin_size, min=0, max=1)

    hist_prob_norm = hist_prob / hist_prob.sum()
    
    hist_prob_perc = [round(1 - item.item(), 3) for item in accumulate(hist_prob_norm)]
    for idx in range(len(hist_prob_perc) - 1, 0, -1):
        if hist_prob_perc[idx] > args.lamda:
            mu = conf_values[idx]
            print("percentage of pixels found:", hist_prob_perc[idx])
            break
    mu_update = mu / (args.n_epochs * args.E_all)
    return mu, mu_update

def train_pixel_wise_curriculum(args, model, teacher_model, criterion, train_loader, valid_loader, validation, num_classes, optimizer):
    seed_everything(1234)
    valid_criterion = CELoss(num_classes = num_classes)
    n_epochs = args.n_epochs
    
    valid_losses = []
    
    if args.method == 'baseline':
        save_model_path = 'saved_model_lr_1e-4/%s/%s' % (args.model, args.method)
    
    elif args.method == 'cbls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f' % (args.model, args.method, args.label_smoothing, args.factor)
    
    elif args.method == 'cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d' % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize)
    
    elif args.method == 'p-cbls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_lamda_%.2f_Eall_%.2f' \
            % (args.model, args.method, args.label_smoothing, args.factor, args.lamda, args.E_all)
    elif args.method == 'p-cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d_lamda_%.2f_Eall_%.2f' \
            % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize, args.lamda, args.E_all)

    elif args.method == 'mixture':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_sig_%.2f_sigfactor_%.2f_ksize_%d_lamda_%.2f_Eall_%.2f' \
            % (args.model, args.method, args.label_smoothing, args.factor, args.sigma, args.sigma_factor, args.ksize, args.lamda, args.E_all)

    # # (Try: For 100 epochs)
    # if args.method == 'p-cbls':
    #     save_model_path = 'old/saved_model_100epochs_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_lamda_%.2f_Eall_%.2f' \
    #         % (args.model, args.method, args.label_smoothing, args.factor, args.lamda, args.E_all)
    # elif args.method == 'p-cbsvls':
    #     save_model_path = 'old/saved_model_100epochs_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d_lamda_%.2f_Eall_%.2f' \
    #         % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize, args.lamda, args.E_all)


    if not os.path.isdir(save_model_path):
        pathlib.Path(save_model_path).mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_epoch = 0
    start_epoch = 0

    if args.resume_last == 'True':
        #------ last model path -------#
        path_checkpoint = 'LinkNet34/baseline/model_epoch_40.pt'
        if os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)

            model.load_state_dict(checkpoint['net'])             # load the model's parameters
            optimizer.load_state_dict(checkpoint['optimizer'])   # load the optimizer's parameters
            start_epoch = checkpoint['epoch'] + 1                # set the start epoch, rememr to add 1
            
            # ========================== Validation ========================== #
            # confirm it is last epoch based on the metric
            valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
            
            # the best_iou in previous experiments is xxx (need to manually update)
            #--- baseline (LinkNer34) ---#
            best_iou =  0.43643259266831347
            best_epoch = 39
            # =================================================================== #
            print('Resuming from epoch %d' % checkpoint['epoch'])

    for epoch in range(start_epoch, n_epochs):

        # update the parameters every epoch
        # ======================= CBLS (decrease) ================== #
        if args.method == 'cbls' or args.method == 'p-cbls':
            if epoch == 0:
                label_smoothing = args.label_smoothing
            elif label_smoothing > 0.0:
                label_smoothing *= args.factor
            else:
                label_smoothing = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' label_smoothing:', label_smoothing)
        # ============================================================ #


        # ======================= CBSVLS (decrease) ================== #
        elif args.method == 'cbsvls' or args.method == 'p-cbsvls':
            if epoch == 0:
                sigma = args.sigma
            elif sigma > 0.0:
                sigma *= args.sigma_factor
            else:
                sigma = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' sigma:', sigma)
        # ============================================================ #

        # ========================= mixture: ================================= #
        # "SVLS on top of ULS": decrease both ULS and SVLS factor during whole training, then we need to combine their loss.
        elif args.method == 'mixture':
            if epoch == 0:
                label_smoothing = args.label_smoothing
                sigma = args.sigma

            elif label_smoothing > 0.0 and sigma > 0.0:
                label_smoothing *= args.factor
                sigma *= args.sigma_factor

                if label_smoothing < 0.0:
                    label_smoothing = 0.0
                if sigma < 0.0:
                    sigma = 0.0
            else:
                label_smoothing = 0.0
                sigma = 0.0
            print('Epoch', epoch, ' label_smoothing:', label_smoothing, ' sigma:', sigma)
        # ==========================================================================#

        model.train()
        teacher_model.eval()

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        # tq.set_description('Epoch {}'.format(epoch))
        
        losses = []
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)


            # # ========= plot the pixel paced learning image =========== # #
            # from loss import get_curriculum_weights_per_px
            # import imageio
            # import cv2
            # from PIL import Image
            # print('plotting !!!!!!!!!!!!!')
            # weights = get_curriculum_weights_per_px(args.num_classes, teacher_outputs, targets, args.mu, args.opt_t)
            # print('weights[0] shape', weights[0].shape) # shape torch.Size([1, 512, 640])
            # # label
            # image = Image.open('/media/mmlab/data/SurgerySegmentaion/sgmt2018/predictions/train_RGB/seq_1_frame000.png')

            # new_size = (640, 512)
            # image = image.resize(new_size)
            # image_array = np.array(image)
            # print('image_array 1:', image_array.shape) # (512, 640, 3)
            # image_array = image_array.transpose(2, 0, 1)
            # pixel_image = (weights[0].cpu()) * image_array 
            # # print('pixel_image 1:', pixel_image.shape) # torch.Size([3, 512, 640])
            # pixel_image = np.array(pixel_image)
            # print('pixel_image 2', pixel_image.shape) # (640, 512, 3) expect it to be (512, 640, 3) pixel_image 2 (3, 512, 640)
            # pixel_image = pixel_image.transpose(1, 2, 0,)
            # print('pixel_image 3', pixel_image.shape)
            # pixel_image = Image.fromarray(pixel_image.astype(np.uint8)).convert('RGB')
            
            # if not os.path.exists('plotted'):
            #     os.makedirs('plotted')
            # saved_name = 'plotted/label_epoch%d.jpg' % epoch
            # pixel_image.save(saved_name)

            # if i == 1:
            #     break       
            # # ========================================================== #
            
            if args.method == 'baseline':
                loss = criterion(outputs, targets) ###############################################
            elif args.method == 'cbls':
                loss = criterion(outputs, targets, label_smoothing)
            elif args.method == 'cbsvls':
                loss = criterion(outputs, targets, sigma)
            elif args.method == 'p-cbls':
                loss = criterion(outputs, teacher_outputs, targets, label_smoothing, args.mu, args.opt_t)
            elif args.method == 'p-cbsvls':
                loss = criterion(outputs, teacher_outputs, targets, sigma, args.mu, args.opt_t)
            
            elif args.method == 'mixture':
                loss = criterion(outputs, teacher_outputs, targets, label_smoothing, sigma, args.mu, args.opt_t)

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()
            
            tq.update(batch_size)
            losses.append(loss.item())

        tq.close()

        # ============= update the mu every epoch for pixel-wise paced learning ==============
        print('epoch:{}, mu:{:.4f}'.format(epoch, args.mu))
        args.mu -= args.mu_update
        args.mu = max(args.mu, 0)
        
        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']
        
        if args.method == 'p-cbls':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "label_smoothing": label_smoothing,
                    "factor": args.factor,
                    "opt_t": args.opt_t,
                    "lamda": args.lamda,
                    "E_all": args.E_all
                }
        elif args.method == 'p-cbsvls':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "sigma": sigma,
                    "sigma_factor": args.sigma_factor,
                    "opt_t": args.opt_t,
                    "lamda": args.lamda,
                    "E_all": args.E_all
                }
        elif args.method == 'mixture':
            checkpoint = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_iou": valid_iou,
                    "best_iou": best_iou,
                    "best_epoch": best_epoch,
                    "valid_loss": valid_loss,
                    "label_smoothing": label_smoothing,
                    "factor": args.factor,
                    "sigma": sigma,
                    "sigma_factor": args.sigma_factor,
                    "opt_t": args.opt_t,
                    "lamda": args.lamda,
                    "E_all": args.E_all
                }            
            
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_epoch = epoch
            print('===================saving the best model!========================')
            # torch.save(model.state_dict(), os.path.join(save_model_path, "best_model.pt"))
            # saved details
            torch.save(checkpoint, os.path.join(save_model_path, "detail_best_model.pt"))
        
        if (epoch+1) % 10 == 0:
            # path_to_save = os.path.join(save_model_path, "model_epoch_%02d.pt" % epoch)
            # torch.save(model.state_dict(), path_to_save)

            # saved details
            path_to_save_detail = os.path.join(save_model_path, "detail_model_epoch_%02d.pt" % epoch)
            torch.save(checkpoint, path_to_save_detail)


        print('best epoch unitl now:', best_epoch, ' best iou unitl now:', best_iou)

    print('best epoch:', best_epoch, ' best iou:', best_iou)


# the last kind of experiments
def train_sample_wise_curriculum(args, model, criterion, train_loader_all, valid_loader, validation, num_classes, optimizer):
    seed_everything(1234)
    valid_criterion = CELoss(num_classes = num_classes)
    
    n_epochs = args.n_epochs
    
    valid_losses = []
    
    if args.method == 'baseline':
        save_model_path = 'saved_model_lr_1e-4/%s/%s' % (args.model, args.method)
    elif args.method == 'cbls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f' % (args.model, args.method, args.label_smoothing, args.factor)
    elif args.method == 'cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d' % (args.model, args.method, args.sigma, args.sigma_factor,args.ksize)

    elif args.method == 'samplewise-p-cbls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_initial_sample_%.2f_all_sample_epoch_ratio_%.2f' \
            % (args.model, args.method, args.label_smoothing, args.factor, args.initial_sample, args.all_sample_epoch_ratio)
    elif args.method == 'samplewise-p-cbsvls':
        save_model_path = 'saved_model_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d_initial_sample_%.2f_all_sample_epoch_ratio_%.2f' \
            % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize, args.initial_sample, args.all_sample_epoch_ratio)


    # if args.method == 'samplewise-p-cbls':
    #     save_model_path = 'old/saved_model_100epochs_lr_1e-4/%s/%s/ls_%.2f_factor_%.2f_initial_sample_%.2f_all_sample_epoch_ratio_%.2f' \
    #         % (args.model, args.method, args.label_smoothing, args.factor, args.initial_sample, args.all_sample_epoch_ratio)
    # elif args.method == 'samplewise-p-cbsvls':
    #     save_model_path = 'old/saved_model_100epochs_lr_1e-4/%s/%s/sig_%.2f_sigfactor_%.2f_ksize_%d_initial_sample_%.2f_all_sample_epoch_ratio_%.2f' \
    #         % (args.model, args.method, args.sigma, args.sigma_factor, args.ksize, args.initial_sample, args.all_sample_epoch_ratio)
    
    if not os.path.isdir(save_model_path):
        pathlib.Path(save_model_path).mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_epoch = 0
    start_epoch = 0 # if training from scratch (not resume training)

    if args.resume_last == 'True':
        #------ last model path -------#
        path_checkpoint = 'LinkNet34/baseline/model_epoch_40.pt'
        if os.path.exists(path_checkpoint):
            checkpoint = torch.load(path_checkpoint)

            model.load_state_dict(checkpoint['net'])           # load the model's parameters
            optimizer.load_state_dict(checkpoint['optimizer']) # load the optimizer's parameters
            start_epoch = checkpoint['epoch'] + 1              # set the start epoch, rememr to add 1
            
            # ========================== Validation ========================== #
            # confirm it is last epoch based on the metric
            valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
            
            # the best_iou in previous experiments is xxx (need to manually update)
            #--- baseline (LinkNer34) ---#
            best_iou =  0.43643259266831347
            best_epoch = 39
            # =================================================================== #
            print('Resuming from epoch %d' % checkpoint['epoch'])

    for epoch in range(start_epoch, n_epochs):

        # update the parameters every epoch
        # ======================= CBLS (decrease) ================== #
        if args.method == 'cbls' or args.method == 'samplewise-p-cbls':
            if epoch == 0:
                label_smoothing = args.label_smoothing
            elif label_smoothing > 0.0:
                label_smoothing *= args.factor
            else:
                label_smoothing = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' label_smoothing:', label_smoothing)
        # ============================================================ #


        # ======================= CBSVLS (decrease) ================== #
        elif args.method == 'cbsvls' or args.method == 'samplewise-p-cbsvls':
            if epoch == 0:
                sigma = args.sigma
            elif sigma > 0.0:
                sigma *= args.sigma_factor
            else:
                sigma = 0.0 # 0.0 is minimum
            print('Epoch', epoch, ' sigma:', sigma)
        # ============================================================ #


         # ''' =======================  Linear MPL =========================== '''
        # num = int((epoch-1) // args.epoch_pace)
        num = int(epoch // args.epoch_pace) # epoch is from 0, not 1 now
        train_loader = train_loader_all[num]
        print('train_loader size', len(train_loader.dataset))
         # ''' ================================================================ '''

        model.train()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))
        # tq.set_description('Epoch {}'.format(epoch))
        
        losses = []
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)

            if args.method == 'baseline':
                loss = criterion(outputs, targets) ###############################################
            elif args.method == 'cbls' or args.method == 'samplewise-p-cbls':
                loss = criterion(outputs, targets, label_smoothing)
            elif args.method == 'cbsvls' or args.method == 'samplewise-p-cbsvls':
                loss = criterion(outputs, targets, sigma)

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()
            
            tq.update(batch_size)
            losses.append(loss.item())

        tq.close()

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, valid_criterion, valid_loader, num_classes)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']


        checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }        
        if valid_iou > best_iou:
            best_iou = valid_iou
            best_epoch = epoch
            print('===================saving the best model!========================')
            torch.save(checkpoint, os.path.join(save_model_path, "detail_best_model.pt"))
        
        if (epoch+1) % 10 == 0:
            path_to_save_detail = os.path.join(save_model_path, "detail_model_epoch_%02d.pt" % epoch)
            torch.save(checkpoint, path_to_save_detail)

    
        # ======================================================= # 
        print('best epoch unitl now:', best_epoch, ' best iou unitl now:', best_iou)

    print('best epoch:', best_epoch, ' best iou:', best_iou)



    # # ========================== mixture 1 ============================ #
    # # 1) "SVLS after ULS": decrease ULS factor during early epochs, then decrease SVLS factor during later epochs.
    # elif args.method == 'mixture_2':
    #     if epoch == 0:
    #         label_smoothing = args.label_smoothing

    #     elif epoch < (n_epochs // 2):
    #         label_smoothing *= args.factor
    #         if label_smoothing < 0.0:
    #             label_smoothing = 0.0

    #     elif epoch == (n_epochs // 2):
    #         print('The key epoch:', epoch)
    #         sigma = args.sigma

    #     elif epoch > (n_epochs // 2):
    #         sigma *= args.sigma_factor
    #         if sigma < 0.0:
    #             sigma = 0.0
    #     print('Epoch', epoch, ' label_smoothing:', label_smoothing, ' sigma:', sigma)
    # # ========================================================================#


        # if args.method =='mixture_2':
    #     if epoch < (n_epochs // 2):
    #         loss = loss_1(outputs, teacher_outputs, targets, label_smoothing, args.mu, args.opt_t)
    #     elif epoch >= (n_epochs // 2):
    #         loss = loss_2(outputs, teacher_outputs, targets, sigma, args.mu, args.opt_t)