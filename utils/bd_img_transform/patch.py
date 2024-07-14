# the callable object for BadNets attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import numpy as np
import torch
from typing import Optional
from torchvision.transforms import Resize, ToTensor, ToPILImage

class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if isinstance(img, np.ndarray):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[m, n, :] = self.trigger_ptn[i]  # add trigger
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n, :] = self.trigger_ptn[i]  # add trigger
        elif isinstance(img, torch.Tensor):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n] = self.trigger_ptn[i]
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, :, m, n] = self.trigger_ptn[i]
        return img

class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)


class AddMaskPatchTrigger_adap_train(object):
    def __init__(self,
                 trigger_list, trigger_alphas, mask_list
                 ):
        self.trigger_list = trigger_list
        self.trigger_alphas = trigger_alphas
        self.mask_list = mask_list
        assert len(trigger_list) == len(trigger_alphas)
        assert len(trigger_list) == len(mask_list)
        self.poi_num = 0
        self.cover_num = 0

    def __call__(self, img, target = None, image_serial_id = None, poi=False):
        return self.add_trigger(img, poi=poi)

    def add_trigger(self, img, poi=False):
        if poi:
            trigger_id = self.poi_num % len(self.trigger_list)
            self.poi_num += 1
        else:
            trigger_id = self.cover_num % len(self.trigger_list)
            self.cover_num += 1
        alpha = self.trigger_alphas[trigger_id]
        trigger = self.trigger_list[trigger_id]
        trigger_mask = self.mask_list[trigger_id]
        # print(np.shape(trigger_mask))
        # print(np.shape(trigger))
        return img * (1-trigger_mask) + alpha * trigger_mask * trigger + (1-alpha) * trigger_mask * img
        # return img + alpha * trigger_mask * (trigger - img)


class AddMaskPatchTrigger_adap_LC_train(object):
    def __init__(self,
                 trigger_list, trigger_alphas, mask_list
                 ):
        self.trigger_list = trigger_list
        self.trigger_alphas = trigger_alphas
        self.mask_list = mask_list
        assert len(trigger_list) == len(trigger_alphas)
        assert len(trigger_list) == len(mask_list)
        self.poi_num = 0

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        trigger_id = self.poi_num % len(self.trigger_list)
        self.poi_num += 1
        alpha = self.trigger_alphas[trigger_id]
        trigger = self.trigger_list[trigger_id]
        trigger_mask = self.mask_list[trigger_id]
        # print(np.shape(trigger_mask))
        # print(np.shape(trigger))
        return img + alpha * trigger_mask * (trigger - img)


class AddMaskPatchTrigger_adap_test(object):
    def __init__(self,
                 trigger_list, trigger_alphas, mask_list
                 ):
        self.trigger_list = trigger_list
        self.trigger_alphas = trigger_alphas
        self.mask_list = mask_list
        assert len(trigger_list) == len(trigger_alphas)
        assert len(trigger_list) == len(mask_list)

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        for j in range(len(self.trigger_list)):
            img = img * (1 - self.mask_list[j]) + self.trigger_alphas[j] * self.mask_list[j] * self.trigger_list[j] \
                  + (1 - self.trigger_alphas[j]) * self.mask_list[j] * img
        return img


class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt
def test_Simple():
    a = SimpleAdditiveTrigger(np.load('../../resource/lowFrequency/cifar10_densenet161_0_255.npy'))
    plt.imshow(a(np.ones((32,32,3)) + 255/2))
    plt.show()
