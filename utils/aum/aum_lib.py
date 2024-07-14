import argparse
import logging
import math
import sys
from copy import deepcopy

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
import subprocess
import platform
import torch
import torch.nn.functional as F

os.chdir(sys.path[0])
sys.path.append('../../')
os.getcwd()
from utils.my_criterion import criterion_partial
from utils.label_smooth import LabelSmoothing_vec
import torch.nn as nn
import pandas as pd
import seaborn as sns


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--save_folder_name", type=str)
    return parser


def density_plot(array_list, label_list, save_path):
    assert len(array_list) == len(label_list)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set_style("white")
    plt.figure()
    # Plot
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:olive', 'tab:pink']
    fig, ax = plt.subplots()
    for i in range(len(array_list)):
        n1, bins, patches1 = plt.hist(array_list[i], bins=50, label=label_list[i],
                                      color=colors[i], alpha=0.6, density=True)
        sigma = math.sqrt(array_list[i].var())
        mu = array_list[i].mean()
        # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
        #      np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
        # plt.plot(bins, y, '-', color=colors[i])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.set_ylabel('Density', fontsize=18)
    ax.set_xlabel('AUM', fontsize=18)
    plt.legend(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)


def get_singleEpoch_prob(dataloader, model, device):
    targets_list, others_list = np.zeros(len(dataloader.dataset)), np.zeros(len(dataloader.dataset))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets, *additional_info) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs, targets
            outputs = model(inputs)
            sf = F.softmax(outputs.data, dim=1)
            ori_index, _, _ = additional_info
            for img_probs, img_idx, img_target in zip(sf.split([1] * len(sf), dim=0),
                                                      ori_index.split([1] * len(ori_index), dim=0),
                                                      targets.split([1] * len(targets), dim=0)):
                img_probs = img_probs.squeeze(dim=0)
                target_pro = img_probs[img_target]
                if img_target < len(img_probs) - 1:
                    notarget_probs = torch.cat([img_probs[:img_target], img_probs[img_target + 1:]], dim=0)
                else:
                    notarget_probs = img_probs[:img_target]
                notarget_probs = notarget_probs.max()
                targets_list[img_idx.item()] = target_pro
                others_list[img_idx.item()] = notarget_probs
    return targets_list, others_list


def loggingAumValue(aum_array, suffix):
    logging.info(f"**************** {suffix} AUM Info ****************")
    lid_mean = aum_array.mean()
    lid_mid = np.median(aum_array)
    logging.info(f"{suffix} AUM mean Value: {lid_mean}")
    logging.info(f"{suffix} AUM mid Value: {lid_mid}")

    # for lid in aum_array:
    #     logging.info(f"{suffix} AUM Value: {lid}")


def loggingSelfEntropyValue(entropy_array, suffix):
    logging.info(f"&&&&&&&&&&&&&&&&& {suffix} Self Entropy Info &&&&&&&&&&&&&&&&&")
    lid_mean = entropy_array.mean()
    lid_mid = np.median(entropy_array)
    logging.info(f"{suffix} Self Entropy mean Value: {lid_mean}")
    logging.info(f"{suffix} Self Entropy mid Value: {lid_mid}")


class AreaUnderTheMarginRanking():
    """
    Implementation of the paper Identifying Mislabeled Data using the Area Under the Margin Ranking: https://arxiv.org/pdf/2001.10528v2.pdf

    Currently the used dataset must not shuffle between epochs !
    """

    # TODO use matrix operations
    # TODO manage the case of dataset with shuffling between epochs
    # TODO Setup full process of filtering a dataset

    def __init__(self, dataloader, model, device):
        # hist_delta_AUM_current_epoch dimensions: [n_sample, 2 (from in_logit & max(out_logits))]
        self.data_loader = dataloader
        self.model = model
        self.device = device
        self.eachEpoch_targets = []
        self.eachEpoch_others = []
        self.accumulated_margin = np.zeros(len(dataloader.dataset))
        self.num_epoch = 0
        self.targets_list, self.others_list = np.zeros(len(dataloader.dataset)), np.zeros(len(dataloader.dataset))
        self.self_know = np.zeros(len(dataloader.dataset))
        self.eachEpoch_SelfEntropy = []
        self.accumulated_selfEntropy = np.zeros(len(dataloader.dataset))

    def oneBatchAccumulateInOneEpoch(self, outputs, targets, ori_index):
        sf = F.softmax(outputs.data, dim=1)
        for img_probs, img_idx, img_target in zip(sf.split([1] * len(sf), dim=0),
                                                  ori_index.split([1] * len(ori_index), dim=0),
                                                  targets.split([1] * len(targets), dim=0)):
            logprobs = F.log_softmax(img_probs, dim=-1)
            self_entropy = -1 * torch.sum(img_probs * logprobs, 1)
            img_probs = img_probs.squeeze(dim=0)
            target_pro = img_probs[img_target]
            if img_target < len(img_probs) - 1:
                notarget_probs = torch.cat([img_probs[:img_target], img_probs[img_target + 1:]], dim=0)
            else:
                notarget_probs = img_probs[:img_target]
            notarget_probs = notarget_probs.max()
            self.targets_list[img_idx.item()] = target_pro
            self.others_list[img_idx.item()] = notarget_probs
            self.self_know[img_idx.item()] = self_entropy

    def oneBatchAccumulateDown(self):
        t_list = self.targets_list
        o_list = self.others_list
        self_know_list = self.self_know
        self.eachEpoch_targets.append(t_list)
        self.eachEpoch_others.append(o_list)
        self.eachEpoch_SelfEntropy.append(self_know_list)
        self.accumulated_margin += t_list - o_list
        self.targets_list, self.others_list = np.zeros(len(self.data_loader.dataset)), np.zeros(
            len(self.data_loader.dataset))
        self.self_know = np.zeros(len(self.data_loader.dataset))
        self.num_epoch += 1
        self.accumulated_selfEntropy += self_know_list

    def epochAccumulate(self):
        self.num_epoch += 1
        targets_list, others_list = get_singleEpoch_prob(self.data_loader, self.model, self.device)
        self.eachEpoch_targets.append(targets_list)
        self.eachEpoch_others.append(others_list)
        self.accumulated_margin += targets_list - others_list

    def get_aumAtCurrentEpoch(self):
        if self.num_epoch == 0:
            return None
        return np.array(self.accumulated_margin / self.num_epoch)

    def get_selfEntropyAtCurrentEpoch(self):
        if self.num_epoch == 0:
            return None
        return np.array(self.accumulated_selfEntropy / self.num_epoch)

    def resetAUM(self):
        self.num_epoch = 0
        self.eachEpoch_targets = []
        self.eachEpoch_others = []

    def get_aumAtCurrentEpochAndRest(self):
        aum = self.get_aumAtCurrentEpoch()
        self.resetAUM()
        return aum

    def get_aumAtCurrentEpochOfTargetClass(self, class_id):
        label_list = self.data_loader.dataset.targets
        p_idx = self.data_loader.dataset.poison_idx
        aum_list = self.get_aumAtCurrentEpoch()
        adv_aum = []
        benign_aum = []
        for i in range(len(label_list)):
            if label_list[i] == class_id:
                if p_idx[i] > 0:
                    adv_aum.append(aum_list[i])
                else:
                    benign_aum.append(aum_list[i])
        return adv_aum, benign_aum

    def get_selfEntropyAtCurrentEpochOfTargetClass(self, class_id):
        label_list = self.data_loader.dataset.targets
        p_idx = self.data_loader.dataset.poison_idx
        se_list = self.get_selfEntropyAtCurrentEpoch()
        adv_se = []
        benign_se = []
        for i in range(len(label_list)):
            if label_list[i] == class_id:
                if p_idx[i] > 0:
                    adv_se.append(se_list[i])
                else:
                    benign_se.append(se_list[i])
        return adv_se, benign_se

    def logging_aumAtCurrentEpochOfTargetClass(self, class_id):
        adv_aum, benign_aum = self.get_aumAtCurrentEpochOfTargetClass(class_id)
        if len(adv_aum) > 0:
            loggingAumValue(np.concatenate((np.array(adv_aum), np.array(benign_aum)), 0),
                            f"total_examples-{self.num_epoch}-label-{class_id}")
            loggingAumValue(np.array(adv_aum), f"poison_examples-{self.num_epoch}-label-{class_id}")
        loggingAumValue(np.array(benign_aum), f"clean_examples-{self.num_epoch}-label-{class_id}")
        return adv_aum, benign_aum

    def logging_SelfEntropyAtCurrentEpochOfTargetClass(self, class_id):
        adv_se, benign_se = self.get_selfEntropyAtCurrentEpochOfTargetClass(class_id)
        if len(adv_se) > 0:
            loggingSelfEntropyValue(np.concatenate((np.array(adv_se), np.array(benign_se)), 0),
                                    f"total_examples-{self.num_epoch}-label-{class_id}")
            loggingSelfEntropyValue(np.array(adv_se), f"poison_examples-{self.num_epoch}-label-{class_id}")
        loggingAumValue(np.array(benign_se), f"clean_examples-{self.num_epoch}-label-{class_id}")
        return adv_se, benign_se

    def plot_SelfEntropyDensitySingleClass(self, class_id, save_path):
        adv_se, benign_se = self.get_selfEntropyAtCurrentEpochOfTargetClass(class_id)
        if len(adv_se) > 0:
            legend_list = ['Benign Samples', 'Poisoned Samples']
            density_plot([np.array(benign_se), np.array(adv_se)], legend_list, save_path)
        else:
            legend_list = ['Benign Samples']
            density_plot([np.array(benign_se)], legend_list, save_path)

    def plot_aumDensitySingleClass(self, class_id, save_path):
        adv_aum, benign_aum = self.get_aumAtCurrentEpochOfTargetClass(class_id)
        if len(adv_aum) > 0:
            legend_list = ['Benign Samples', 'Poisoned Samples']
            density_plot([np.array(benign_aum), np.array(adv_aum)], legend_list, save_path)
        else:
            legend_list = ['Benign Samples']
            density_plot([np.array(benign_aum)], legend_list, save_path)

    def logging_selfEntropyAtCurrentEpoch(self):
        self_e = self.get_selfEntropyAtCurrentEpoch()
        adv_e = []
        clean_e = []
        p_idx = self.data_loader.dataset.poison_idx
        for i in range(len(p_idx)):
            if p_idx[i] > 0:
                adv_e.append(self_e[i])
            else:
                clean_e.append(self_e[i])

        loggingSelfEntropyValue(np.array(adv_e), f"poison_examples-{self.num_epoch}")
        loggingSelfEntropyValue(np.array(clean_e), f"poison_examples-{self.num_epoch}")
        loggingSelfEntropyValue(self_e, f"total_examples-{self.num_epoch}")

    def logging_aumAtCurrentEpoch(self):
        aum = self.get_aumAtCurrentEpoch()
        adv_aum = []
        clean_aum = []
        p_idx = self.data_loader.dataset.poison_idx
        for i in range(len(p_idx)):
            if p_idx[i] > 0:
                adv_aum.append(aum[i])
            else:
                clean_aum.append(aum[i])
        loggingAumValue(np.array(adv_aum), f"poison_examples-{self.num_epoch}")
        loggingAumValue(np.array(clean_aum), f"clean_examples-{self.num_epoch}")
        loggingAumValue(aum, f"total_examples-{self.num_epoch}")
        self.logging_selfEntropyAtCurrentEpoch()

    def getList_aumBiggerThan(self, rate):
        aum = self.get_aumAtCurrentEpoch()
        target_array = np.zeros(len(self.data_loader.dataset))
        for i, value in enumerate(aum):
            if value > rate:
                target_array[i] = 1
        return target_array.tolist()

    def getList_aumBiggerThanMean(self):
        aum_mean = self.get_aumAtCurrentEpoch().mean()
        return self.getList_aumBiggerThan(aum_mean)


def get_FinalAUM(args, dataset, model, device):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    aum_accumulator = AreaUnderTheMarginRanking(dataloader, model, device)
    # aum_accumulator.epochAccumulate()
    # aum_accumulator.logging_aumAtCurrentEpoch()

    model.eval()
    criterion = criterion_partial(par_rate=0.01)
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets, *additional_info) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs, targets
            outputs = model(inputs)
            ori_index, poison_indicator, _ = additional_info
            loss = criterion(outputs, targets.long(), poison_indicator)

            aum_accumulator.oneBatchAccumulateInOneEpoch(outputs, targets, ori_index)

    aum_accumulator.oneBatchAccumulateDown()
    # aum_accumulator.logging_aumAtCurrentEpoch()

    # locate save director
    plat = platform.system().lower()
    # save representations
    # win
    if plat == 'windows':
        absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # linux
    else:
        absolute_path = '/mnt/BackdoorBench-main'
    save_dir = os.path.join(absolute_path, args.save_path[3:], 'aum_density')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    aum_accumulator.plot_aumDensitySingleClass(0, save_dir + '/label-0_aum_density.pdf')
    aum_accumulator.plot_SelfEntropyDensitySingleClass(0, save_dir + '/label-0_se_density.pdf')


def path_test(save_path):
    absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # print(os.path.dirname(__file__))
    # print(absolute_path)
    # print(os.path.dirname(absolute_path))
    # print(os.path.dirname(os.path.dirname(absolute_path)))
    absolute_path = '/mnt/BackdoorBench-main'
    quantum_path = os.path.join(absolute_path, save_path[3:], 'quantum')
    print(quantum_path)
