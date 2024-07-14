import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class LabelSmoothing_vec(nn.Module):
    def __init__(self):
        super(LabelSmoothing_vec, self).__init__()

    def forward(self, prediction, target):
        logprobs = F.log_softmax(prediction, dim=-1)
        loss = -1 * torch.sum(target * logprobs, 1)
        return loss.mean()


class LabelSmoothing_dynamic(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, class_num=10):
        super().__init__()
        self.class_num = class_num

    def forward(self, pred, target, label_smooth=None):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        t = target
        if label_smooth is not None:
            # cross entropy loss with label smoothing
            ls = (label_smooth/self.class_num)*(self.class_num-1)
            logprobs = F.log_softmax(pred, dim=1)
            # if len(target) > 1:
            #     target_list = []
            #     for i in range(len(target)):
            #         one_hot = torch.full_like(pred[i], fill_value=ls[i] / (self.class_num - 1), dtype=torch.float)
            #         one_hot[target[i]] = 1.0 - ls[i]
            #         target_list.append(one_hot)
            #     target = torch.stack((tuple(target_list)))
            # else:
            #     one_hot = torch.full_like(pred, fill_value=ls / (self.class_num - 1), dtype=torch.float)
            #     one_hot[target] = 1.0
            #     target = one_hot
            # loss = -1 * torch.sum(target * logprobs, 1)
            confidence = 1. - label_smooth
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = confidence * nll_loss + label_smooth * smooth_loss

        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        return loss.mean()


class LabelSmoothing_target(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=10):
        super().__init__()
        self.class_num = class_num
        self.label_smooth = (label_smooth/self.class_num)*(self.class_num-1)

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12
        if self.label_smooth is not None:
            # cross entropy loss with label smoothing
            logprobs = F.log_softmax(pred, dim=1)
            if len(target) > 1:
                target_list = []
                for i in range(len(target)):
                    one_hot = torch.full_like(pred[i], fill_value=self.label_smooth/ (self.class_num - 1), dtype=torch.float)
                    one_hot[target[i]] = 1.0 - self.label_smooth
                    target_list.append(one_hot)
                target = torch.stack((tuple(target_list)))
            else:
                one_hot = torch.full_like(pred, fill_value=self.label_smooth / (self.class_num - 1), dtype=torch.float)
                one_hot[target] = 1.0
                target = one_hot
            loss = -1 * torch.sum(target * logprobs, 1)
        else:
            # standard cross entropy loss
            loss = -1. * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        return loss.mean()


if __name__ == '__main__':
    x = torch.tensor([[1, 8, 1], [1, 1, 8]], dtype=torch.float)
    y = torch.tensor([[0.025, 0.95, 0.025], [0.025, 0.025, 0.95]])
    y2 = torch.tensor([1, 2])

    criterion = LabelSmoothing_vec()
    CELoss = LabelSmoothing_target(0.05, 3)

    print(criterion(x, y))
    print(CELoss(x, y2))


