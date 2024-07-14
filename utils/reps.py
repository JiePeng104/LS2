import sys
import numpy as np
import pandas as pd
import scipy
import scipy.sparse.linalg
import torch
import tqdm
from functools import partial
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Collection, Dict, List, Union
import torch.backends.cudnn as cudnn


def compute_all_reps(
    model: torch.nn.Sequential,
    dataloader,
    dataset,
    layers: Collection[int],
    flat=False,
) -> Dict[int, np.ndarray]:
    device = "cpu"
    n = len(dataset)
    max_layer = max(layers)
    # assert max_layer < len(model)

    reps = {}
    x = dataset[0][0][None, ...].to(device)
    for i, layer in enumerate(model):
        if i > max_layer:
            break
        x = layer(x)
        if i in layers:
            inner_shape = x.shape[1:]
            reps[i] = torch.empty(n, *inner_shape)

    print('hi')

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            for i, layer in enumerate(model):
                if i > max_layer:
                    break
                x = layer(x)
                if i in layers:
                    reps[i][start_index : start_index + minibatch_size] = x.cpu()

            start_index += minibatch_size

    if flat:
        for layer in reps:
            layer_reps = reps[layer]
            reps[layer] = layer_reps.reshape(layer_reps.shape[0], -1)

    return reps