'''
this script is for badnet attack

basic structure:
1. config args, save_path, fix random seed
2. set the clean train data and clean test data
3. set the attack img transform and label transform
4. set the backdoor attack data and backdoor test data
5. set the device, model, criterion, optimizer, training schedule.
6. attack or use the model to do finetune with 5% clean data
7. save the attack result for defense
'''

import os
import sys
import yaml

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import pformat
import numpy as np
import torch
import time
import logging

from utils.aggregate_block.save_path_generate import generate_save_folder
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion, \
    argparser_criterion2
from utils.save_load_attack import save_attack_result, load_attack_result
from utils.log_assist import get_git_info


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--patch_mask_path", type=str)
    parser.add_argument('--amp', type=lambda x: str(x) in ['True', 'true', '1'])
    parser.add_argument('--device', type=str)
    parser.add_argument('--attack', type=str, )
    parser.add_argument('--yaml_path', type=str, default='../config/attack/badnet/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')
    parser.add_argument('--load_path', type=str,
                        help='record_path')

    parser.add_argument('--dynamic_ls', type=bool, default=False,
                        help='dynamic_ls')
    parser.add_argument('--nls', type=bool, default=False,
                        help='negative ls')

    parser.add_argument('--p_rate', type=float, default=0.2,
                        help='dynamic_label smooth : positive rate')
    parser.add_argument('--n_rate', type=float, default=0.0,
                        help='dynamic_label smooth : negative rate')
    parser.add_argument('--warmup_end', type=int, default=10,
                        help='the epoch to end warmup procedure and start dynamic label smooth')
    parser.add_argument('--warmup_rate', type=float, default=0.1,
                        help='label smooth rate during warmup')
    parser.add_argument('--log_aum', type=bool, default=False, help='logging AUM')
    parser.add_argument('--smooth_rate', type=float, default=0.0, help='label smooth rate')

    return parser


def main():
    ### 1. config args, save_path, fix random seed
    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k: v for k, v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv

    args.num_classes = get_num_classes(args.dataset)
    args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"

    ### save path
    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')

    ### set the logger
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    try:
        logging.info(pformat(get_git_info()))
    except:
        logging.info('Getting git info fails.')

    ### set the random seed
    fix_random(int(args.random_seed))

    ### 2. set the clean train data and clean test data
    train_dataset_without_transform, \
    train_img_transform, \
    train_label_transfrom, \
    test_dataset_without_transform, \
    test_img_transform, \
    test_label_transform = dataset_and_transform_generate(args)

    benign_train_ds = prepro_cls_DatasetBD(
        full_dataset_without_transform=train_dataset_without_transform,
        poison_idx=np.zeros(len(train_dataset_without_transform)),
        # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
    )

    benign_test_ds = prepro_cls_DatasetBD(
        test_dataset_without_transform,
        poison_idx=np.zeros(len(test_dataset_without_transform)),
        # one-hot to determine which image may take bd_transform
        bd_image_pre_transform=None,
        bd_label_pre_transform=None,
        ori_image_transform_in_loading=test_img_transform,
        ori_label_transform_in_loading=test_label_transform,
        add_details_in_preprocess=True,
    )

    ### 3. set the attack img transform and label transform
    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)
    ### get the backdoor transform on label
    bd_label_transform = bd_attack_label_trans_generate(args)

    ### 4. set the backdoor attack data and backdoor test data
    train_pidx = generate_pidx_from_label_transform(
        benign_train_ds.targets,
        label_transform=bd_label_transform,
        train=True,
        pratio=args.pratio if 'pratio' in args.__dict__ else None,
        p_num=args.p_num if 'p_num' in args.__dict__ else None,
    )

    torch.save(train_pidx,
               args.save_path + '/train_pidex_list.pickle',
               )

    ### generate train dataset for backdoor attack
    adv_train_ds = prepro_cls_DatasetBD(
        deepcopy(train_dataset_without_transform),
        poison_idx=train_pidx,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
    )

    ### decide which img to poison in ASR Test
    test_pidx = generate_pidx_from_label_transform(
        benign_test_ds.targets,
        label_transform=bd_label_transform,
        train=False,
    )

    ### generate test dataset for ASR
    adv_test_dataset = prepro_cls_DatasetBD(
        deepcopy(test_dataset_without_transform),
        poison_idx=test_pidx,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=test_img_transform,
        ori_label_transform_in_loading=test_label_transform,
        add_details_in_preprocess=True,
    )

    # delete the samples that do not used for ASR test (those non-poisoned samples)
    adv_test_dataset.subset(
        np.where(test_pidx == 1)[0]
    )

    ### 5. set the device, model, criterion, optimizer, training schedule.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.img_size[0],
    )

    if torch.cuda.device_count() > 1 and args.device == 'cuda':
        logging.info("device='cuda', default use all device")
        net = torch.nn.DataParallel(net)

    trainer = generate_cls_trainer(
        net,
        args.attack,
        args.amp,
    )

    criterion = argparser_criterion2(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)
    logging.info(f'logAUM1:{args.log_aum}')

    ### 6. attack or use the model to do finetune with 5% clean data
    if 'load_path' not in args.__dict__:
        logging.info(f'logAUM2:{args.log_aum}')

        trainer.train_with_test_each_epoch_v2_sp(
            batch_size=args.batch_size,
            train_dataset=adv_train_ds,
            test_dataset_dict={
                "test_data": benign_test_ds,
                "adv_test_data": adv_test_dataset,
            },
            end_epoch_num=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            frequency_save=args.frequency_save,
            save_folder_path=save_path,
            save_prefix='attack',
            continue_training_path=None,
            dynamic_ls=args.dynamic_ls,
            p_rate=args.p_rate,
            n_rate=args.n_rate,
            warmup_end=args.warmup_end,
            warmup_rate=args.warmup_rate,
            nls=args.nls,
            log_aum=args.log_aum,
            args=args
        )

    else:
        model_name = args.model

        net = generate_cls_model(
            model_name=model_name,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        result = load_attack_result('../record/' + args.load_path + '/attack_result.pt')
        net.load_state_dict(result['model'])

        optimizer, scheduler = argparser_opt_scheduler(net, args)

        trainer = generate_cls_trainer(
            net,
            args.attack,
            args.amp,
        )

        trainer.train_with_test_each_epoch_v2_sp(
            batch_size=args.batch_size,
            train_dataset=adv_train_ds,
            test_dataset_dict={
                "test_data": benign_test_ds,
                "adv_test_data": adv_test_dataset,
            },
            end_epoch_num=args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            frequency_save=args.frequency_save,
            save_folder_path=save_path,
            save_prefix='attack',
            continue_training_path=None,
            dynamic_ls=args.dynamic_ls,
            p_rate=args.p_rate,
            n_rate=args.n_rate,
            warmup_end=args.warmup_end,
            warmup_rate=args.warmup_rate,
            nls=args.nls
        )

    ### 7. save model, data, and other information that defense process may need
    save_attack_result(
        model_name=args.model,
        num_classes=args.num_classes,
        model=trainer.model.cpu().state_dict(),
        data_path=args.dataset_path,
        img_size=args.img_size,
        clean_data=args.dataset,
        bd_train=adv_train_ds,
        bd_test=adv_test_dataset,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
