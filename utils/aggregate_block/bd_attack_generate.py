# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.
import os
import sys, logging


sys.path.append('../../')
import  imageio
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.blended import blendedImageAttack, blendedImageAttack_adap_train
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger, AddMaskPatchTrigger_adap_train, \
    AddMaskPatchTrigger_adap_test, AddMaskPatchTrigger_adap_LC_train
from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.sig import sigTriggerAttack, Segment_sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_img_transform.ft_trojan import FtTrojanAttack
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img


def get_trigger_mask(args, trigger_name, trans):
    trigger_list = []
    mask_list = []
    trigger_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    np_trans = transforms.Compose([
        np.array,
    ])

    re_size_trans = transforms.Compose([
        trans.transforms[1]
    ])

    for trigger in trigger_name:
        from PIL import Image
        trigger = Image.open(os.path.join(args.patch_mask_path, trigger)).convert("RGB")
        trigger = trigger_transform(trigger)
        trigger_list.append(trans(trigger))
        t_trigger = re_size_trans(trigger)

        # load mask
        trigger_mask_path = os.path.join(args.patch_mask_path, 'mask_%s' % trigger)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            # TODO
            # trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
            trigger_mask = trans(trigger_transform(trigger_mask))[0]
            print(f"exist: {np.shape(trigger_mask)}")

        else:  # by default, all black pixels are masked with 0's
            import torch
            trigger_mask = torch.logical_or(torch.logical_or(t_trigger[0] > 0, t_trigger[1] > 0),
                                            t_trigger[2] > 0).float()
            trigger_mask = np_trans(trigger_mask).astype(np.uint8)
            print(f"not exist: {np.shape(trigger_mask)}, p-{trigger_mask_path}")
            logging.info(f"not exist: {np.shape(trigger_mask)}, p-{trigger_mask_path}")

        trigger_mask = np.array(trigger_mask)
        mask_list.append(np.array((trigger_mask, trigger_mask, trigger_mask)).transpose(1, 2, 0))

    return trigger_list, mask_list


def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack == 'fix_patch':

        # trigger_loc = args.attack_trigger_loc # [[26, 26], [26, 27], [27, 26], [27, 27]]
        # trigger_ptn = args.trigger_ptn # torch.randint(0, 256, [len(trigger_loc)])
        # bd_transform = AddPatchTrigger(
        #     trigger_loc=trigger_loc,
        #     trigger_ptn=trigger_ptn,
        # )

        if not os.path.exists(args.patch_mask_path):
            tmp = args.patch_mask_path
            args.patch_mask_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(np.load(args.patch_mask_path)),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
        ])

    elif args.attack == 'adap_patch':

        # trigger_loc = args.attack_trigger_loc # [[26, 26], [26, 27], [27, 26], [27, 27]]
        # trigger_ptn = args.trigger_ptn # torch.randint(0, 256, [len(trigger_loc)])
        # bd_transform = AddPatchTrigger(
        #     trigger_loc=trigger_loc,
        #     trigger_ptn=trigger_ptn,
        # )

        if not os.path.exists(args.patch_mask_path):
            tmp = args.patch_mask_path
            args.patch_mask_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        trigger_name = ['phoenix_corner_32.png', 'firefox_corner_32.png',
                        'badnet_patch4_32.png', 'trojan_square_32.png']
        trigger_alphas = [0.5, 0.2, 0.5, 0.3]
        trigger_list, mask_list = get_trigger_mask(args, trigger_name, trans)

        train_bd_transform = AddMaskPatchTrigger_adap_train(
            trigger_list, trigger_alphas, mask_list
        )

        trigger_name_test = ['phoenix_corner_32.png', 'badnet_patch4_32.png']
        trigger_alphas_test = [1, 1]
        trigger_list_test, mask_list_test = get_trigger_mask(args, trigger_name_test, trans)

        test_bd_transform = AddMaskPatchTrigger_adap_test(
            trigger_list_test, trigger_alphas_test, mask_list_test
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (train_bd_transform, True),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (test_bd_transform, True),
        ])

    elif args.attack == 'blended':

        if not os.path.exists(args.attack_trigger_img_path):
            tmp = args.attack_trigger_img_path
            args.attack_trigger_img_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha)), True) # 0.1,
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)), True) # 0.1,
        ])

    elif args.attack == 'blended_adap':

        if not os.path.exists(args.attack_trigger_img_path):
            tmp = args.attack_trigger_img_path
            args.attack_trigger_img_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack_adap_train(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_train_blended_alpha), img_size=args.img_size[0]), True) # 0.1,
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (blendedImageAttack(
            trans(
                imageio.imread(args.attack_trigger_img_path) # '../data/hello_kitty.jpeg'
                  ).cpu().numpy().transpose(1, 2, 0) * 255,
            float(args.attack_test_blended_alpha)), True) # 0.1,
        ])
        
    elif args.attack == 'ft_trojan':
    
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]), # (32, 32)
            transforms.ToTensor()
        ])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (FtTrojanAttack(args.yuv_flag, args.window_size, args.pos_list, args.magnitude), False),
        ])
        
    elif args.attack == 'sig':
        trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (trans, True),
        ])

    elif args.attack == 'adap_sig':
        train_trans = Segment_sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
            img_size=args.img_size[0],
            mask_rate=args.mask_rate
        )
        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (train_trans, True),
        ])
        test_trans = sigTriggerAttack(
            delta=args.sig_delta,
            f=args.sig_f,
        )
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (test_trans, True),
        ])

    elif args.attack == 'adap_lc':
        if not os.path.exists(args.attack_train_replace_imgs_path):
            tmp = args.attack_train_replace_imgs_path
            args.attack_train_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])
        if not os.path.exists(args.attack_test_replace_imgs_path):
            tmp = args.attack_test_replace_imgs_path
            args.attack_test_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        if not os.path.exists(args.patch_mask_path):
            tmp = args.patch_mask_path
            args.patch_mask_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        trigger_name = ['phoenix_corner_32.png', 'firefox_corner_32.png',
                        'badnet_patch4_32.png', 'trojan_square_32.png']
        trigger_alphas = [0.5, 0.2, 0.5, 0.3]
        trigger_list, mask_list = get_trigger_mask(args, trigger_name, trans)

        train_bd_transform_trigger = AddMaskPatchTrigger_adap_LC_train(
            trigger_list, trigger_alphas, mask_list
        )

        trigger_name_test = ['phoenix_corner_32.png', 'badnet_patch4_32.png']
        trigger_alphas_test = [1, 1]
        trigger_list_test, mask_list_test = get_trigger_mask(args, trigger_name_test, trans)

        test_bd_transform_trigger = AddMaskPatchTrigger_adap_test(
            trigger_list_test, trigger_alphas_test, mask_list_test
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_train_replace_imgs_path)['data'] # '../data/cifar10_SSBA/train.npy'
                ), True),
            (train_bd_transform_trigger, True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_test_replace_imgs_path)['data'] #'../data/cifar10_SSBA/test.npy'
                ),True),
            (test_bd_transform_trigger, True),
        ])

    elif args.attack in ['SSBA_replace']:
        if not os.path.exists(args.attack_train_replace_imgs_path):
            tmp = args.attack_train_replace_imgs_path
            args.attack_train_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        if not os.path.exists(args.attack_test_replace_imgs_path):
            tmp = args.attack_test_replace_imgs_path
            args.attack_test_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_train_replace_imgs_path) # '../data/cifar10_SSBA/train.npy'
                ), True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_test_replace_imgs_path) #'../data/cifar10_SSBA/test.npy'
                ),True),
        ])

    elif args.attack in ['label_consistent']:
        if not os.path.exists(args.attack_train_replace_imgs_path):
            tmp = args.attack_train_replace_imgs_path
            args.attack_train_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])
        if not os.path.exists(args.attack_test_replace_imgs_path):
            tmp = args.attack_test_replace_imgs_path
            args.attack_test_replace_imgs_path = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        add_trigger = labelConsistentAttack("all-corners", reduced_amplitude=1)
        add_trigger_func = add_trigger.poison_from_indices

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_train_replace_imgs_path)['data'] # '../data/cifar10_SSBA/train.npy'
                ), True),
            (add_trigger_func, False),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SSBA_attack_replace_version(
            replace_images=np.load(args.attack_test_replace_imgs_path)['data'] #'../data/cifar10_SSBA/test.npy'
                ),True),
            (add_trigger_func, False),
        ])

    elif args.attack == 'lowFrequency':
        if not os.path.exists(args.lowFrequencyPatternPath):
            tmp = args.lowFrequencyPatternPath
            args.lowFrequencyPatternPath = os.path.join(os.getcwd(), tmp.split('/')[1], tmp.split('/')[2], tmp.split('/')[3])

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = np.load(args.lowFrequencyPatternPath)
            ), True),
        ])
        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (SimpleAdditiveTrigger(
                trigger_array = np.load(args.lowFrequencyPatternPath)
            ), True),
        ])

    return train_bd_transform, test_bd_transform

def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount), int(args.num_classes)
        )

    return bd_label_transform

