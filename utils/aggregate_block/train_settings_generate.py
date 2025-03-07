# This script contains function to set the training criterion, optimizer and schedule.

import sys, logging
sys.path.append('../../')
import torch
import torch.nn as nn
from utils.label_smooth import LabelSmoothing_target, LabelSmoothing_dynamic
from utils.sam_tool.sam import SAM


class flooding(torch.nn.Module):
    # idea: module that can add flooding formula to the loss function
    '''The additional flooding trick on loss'''
    def __init__(self, inner_criterion, flooding_scalar = 0.5):
        super(flooding, self).__init__()
        self.inner_criterion = inner_criterion
        self.flooding_scalar = float(flooding_scalar)
    def forward(self, output, target):
        return (self.inner_criterion(output, target) - self.flooding_scalar).abs() + self.flooding_scalar

def argparser_criterion(args):
    '''
    # idea: generate the criterion, default is CrossEntropyLoss
    '''
    criterion = nn.CrossEntropyLoss()
    if ('flooding_scalar' in args.__dict__): # use the flooding formulation warpper
        criterion = flooding(
            criterion,
            flooding_scalar=float(
                            args.flooding_scalar
                        )
        )
    return criterion


def argparser_criterion2(args):
    '''
    # generate the criterion, default is CrossEntropyLoss
    # when smooth_rate is not 0, output the Label Smoothing criterion
    '''
    if ('smooth_rate' in args.__dict__) and not (args.smooth_rate == 0.0):
        nc = args.num_classes
        if args.dynamic_ls:
            criterion = LabelSmoothing_dynamic(class_num=nc)
            logging.info('Dynamic LS')
        else:
            criterion = LabelSmoothing_target(label_smooth=args.smooth_rate, class_num=nc)
            logging.info('Standard LS')
    else:
        criterion = nn.CrossEntropyLoss()
        if ('flooding_scalar' in args.__dict__):  # use the flooding formulation warpper
            criterion = flooding(
                criterion,
                flooding_scalar=float(
                    args.flooding_scalar
                )
            )
        logging.info('Cross Entropy')
    return criterion
#
# def argparser_criterion2(args):
#     '''
#     # idea: generate the criterion, default is CrossEntropyLoss
#     '''
#     if ('smooth_rate' in args.__dict__): # use the flooding formulation warpper
#         nc = args.num_classes
#         if args.dynamic_ls:
#             criterion = LabelSmoothing_dynamic(class_num=nc)
#         else:
#             criterion = LabelSmoothing_target(label_smooth=args.smooth_rate, class_num=nc)
#     else:
#         criterion = nn.CrossEntropyLoss()
#         if ('flooding_scalar' in args.__dict__): # use the flooding formulation warpper
#             criterion = flooding(
#                 criterion,
#                 flooding_scalar=float(
#                                 args.flooding_scalar
#                             )
#             )
#     return criterion


def argparser_opt_scheduler_poi(model, args):
    # idea: given model and args, return the optimizer and scheduler you choose to use

    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.fix_lr,
                                    momentum=args.sgd_momentum,  # 0.9
                                    weight_decay=args.wd,  # 5e-4
                                    )
    # Sharpness-Aware min
    elif args.client_optimizer == 'sam_train':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.fix_lr, rho=0.05,
                        momentum=args.sgd_momentum,  # 0.9
                        weight_decay=args.wd)

    elif args.client_optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.fix_lr,
            rho = args.rho, #0.95,
            eps = args.eps, #1e-07,
        )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.fix_lr,
                                     betas=args.adam_betas,
                                     weight_decay=args.wd,
                                     amsgrad=True)

    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.min_lr,
                                                      max_lr=args.fix_lr,
                                                      step_size_up= args.step_size_up,
                                                      step_size_down= args.step_size_down,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.steplr_stepsize,  # 1
                                                    gamma=args.steplr_gamma)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplr_milestones ,args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                'factor':args.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in args.__dict__ else {})
        )
    else:
        scheduler = None

    return optimizer, scheduler


def argparser_opt_scheduler(model, args):
    # idea: given model and args, return the optimizer and scheduler you choose to use

    if args.client_optimizer == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=args.sgd_momentum,  # 0.9
                                    weight_decay=args.wd,  # 5e-4
                                    )
    # Sharpness-Aware min
    elif args.client_optimizer == 'sam_train':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.1,
                        momentum=args.sgd_momentum,  # 0.9
                        weight_decay=args.wd)

    elif args.client_optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = args.lr,
            rho = args.rho, #0.95,
            eps = args.eps, #1e-07,
        )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.lr,
                                     betas=args.adam_betas,
                                     weight_decay=args.wd,
                                     amsgrad=True)

    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.min_lr,
                                                      max_lr=args.lr,
                                                      step_size_up= args.step_size_up,
                                                      step_size_down= args.step_size_down,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.steplr_stepsize,  # 1
                                                    gamma=args.steplr_gamma)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplr_milestones ,args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                'factor':args.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in args.__dict__ else {})
        )
    else:
        scheduler = None

    return optimizer, scheduler
