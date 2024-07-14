import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from torch.autograd import Variable

class criterion_partial(nn.Module):
    ''' Cross Entropy Loss with Different weight for Different Sample'''

    def __init__(self, par_rate=0.01, criterion=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.criterion = criterion
        self.par_rate = par_rate

    def forward(self, pred, target, par_index):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        par_index = np.array(par_index)
        loss = self.criterion(pred, target)
        for i, value in enumerate(par_index):
            if value > 0:
                loss[i] = loss[i]*self.par_rate
        return loss


class omega_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, rand_idx):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        logit = pred[rand_idx]
        label_idx = target[rand_idx]
        loss = logit[label_idx]-logit[label_idx]
        for j in range(len(logit)):
            loss += logit[label_idx]-logit[j]
        return loss


class criterion_partial2(nn.Module):
    ''' Cross Entropy Loss with Different weight for Different Sample'''

    def __init__(self, par_rate=0.01, criterion=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.criterion = criterion
        self.par_rate = par_rate

    def forward(self, pred, target, par_index, device='cuda:0'):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        par_index = np.array(par_index)
        loss = self.criterion(pred, target)
        poi_index, cl_index = [], []
        for i, value in enumerate(par_index):
            if value > 0:
                poi_index.append(i)
            else:
                cl_index.append(i)
        cl_loss = torch.index_select(loss, 0, torch.tensor(cl_index).to(device))
        poi_loss = torch.index_select(loss, 0, torch.tensor(poi_index).to(device))
        return (cl_loss.mean() + poi_loss.mean()*self.par_rate) / 2


if __name__ == '__main__':
    x = torch.tensor([[1, 8, 1], [1, 1, 8], [1, 2, 6]], dtype=torch.float)
    x2 = torch.tensor([[0.4, 0.1, -0.1], [0.2, 0.2, 0.1], [0.1, -0.2, 0.5]], dtype=torch.float)
    y = torch.tensor([[0.025, 0.95, 0.025], [0.025, 0.025, 0.95], [0.03, 0.03, 0.94]])
    y2 = torch.tensor([1, 2, 2])

    criterion5 = omega_loss()
    print(criterion5(x2, y2, 0))

    # criterion1 = criterion_partial()
    # criterion2 = criterion_partial(1)
    # criterion3 = criterion_partial2()
    # criterion4 = criterion_partial2(1)
    #
    # CELoss = nn.CrossEntropyLoss()
    #
    # print(CELoss(y, y2))
    # print(criterion1(y, y2, [0, 1, 0]).mean())
    # print(criterion2(y, y2, [0, 1, 0]).mean())
    # print(criterion3(y, y2, [0, 1, 0]).mean())
    # print(criterion4(y, y2, [0, 1, 0]).mean())


