import numpy as np


# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
class blendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1 - self.blended_rate) * img + (self.blended_rate) * self.target_image


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


class blendedImageAttack_adap_train(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate, pieces=16, mask_rate=0.5, img_size=32):
        self.target_image = target_image
        self.blended_rate = blended_rate
        self.mask_rate = mask_rate
        self.pieces = pieces
        self.masked_pieces = round(self.mask_rate * self.pieces)
        self.img_size = img_size

    def __call__(self, img, target=None, image_serial_id=None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        mask = get_trigger_mask(self.img_size, self.pieces, self.masked_pieces)
        mask = np.array(mask)
        mask_rgb = np.array((mask, mask, mask)).transpose(1, 2, 0)

        return img + self.blended_rate * mask_rgb * (self.target_image - img)
