#This script is for Sig attack callable transform

'''
This code is based on https://github.com/bboylyg/NAD

The original license:
License CC BY-NC

The update include:
    1. change to callable object
    2. change the way of trigger generation, use the original formulation.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
'''

from typing import Union
import torch
import numpy as np


def get_trigger_mask(img_size, total_pieces, masked_pieces):
    from math import sqrt
    import torch
    import random
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask


class sigTriggerAttack(object):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    def __init__(self,
                 delta : Union[int, float, complex, np.number, torch.Tensor] = 40,
                 f : Union[int, float, complex, np.number, torch.Tensor] =6
                 ) -> None:

        self.delta = delta
        self.f = f

    def __call__(self, img, target = None, image_serial_id = None):
        return self.sigTrigger(img)

    def sigTrigger(self, img):

        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
              for j in range(int(img.shape[1])):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))

        return img

class Segment_sigTriggerAttack(object):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    def __init__(self,
                 delta : Union[int, float, complex, np.number, torch.Tensor] = 40,
                 f : Union[int, float, complex, np.number, torch.Tensor] =6,
                 pieces=16, mask_rate=0.5,
                 img_size=32,
                 ) -> None:
        self.delta = delta
        self.f = f
        self.img_size = img_size
        self.pieces = pieces
        self.masked_pieces = round(mask_rate * self.pieces)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.sigTrigger(img)

    def sigTrigger(self, img):
        mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
        mask = np.array(mask)
        mask_rgb = np.array((mask, mask, mask)).transpose(1, 2, 0)

        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
              for j in range(int(img.shape[1])):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        img = np.uint32(img) + mask_rgb * pattern
        img = np.uint8(np.clip(img, 0, 255))

        return img


