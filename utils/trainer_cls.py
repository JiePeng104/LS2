# This script is for trainer. This is a warpper for training process.

import sys, logging
from tqdm import tqdm

sys.path.append('../')
import random
from pprint import pformat
from collections import deque
from typing import *
import numpy as np
import torch
import pandas as pd
from time import time
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from utils.aum.aum_lib import AreaUnderTheMarginRanking, density_plot


class dl_generator:
    def __init__(self, **kwargs_init):
        self.kwargs_init = kwargs_init

    def __call__(self, *args, **kwargs_call):
        kwargs = deepcopy(self.kwargs_init)
        kwargs.update(kwargs_call)
        return DataLoader(
            *args,
            **kwargs
        )


def last_and_valid_max(col: pd.Series):
    '''
    find last not None value and max valid (not None or np.nan) value for each column
    :param col:
    :return:
    '''
    return pd.Series(
        index=[
            'last', 'valid_max', 'exist_nan_value'
        ],
        data=[
            col[~col.isna()].iloc[-1], pd.to_numeric(col, errors='coerce').max(), any(i == 'nan_value' for i in col)
        ])


class Metric_Aggregator(object):
    '''
    aggregate the metric to log automatically
    '''

    def __init__(self):
        self.history = []

    def __call__(self,
                 one_metric: dict):
        one_metric = {k: v for k, v in one_metric.items() if v is not None}  # drop pair with None as value
        one_metric = {
            k: (
                "nan_value" if v is np.nan or torch.tensor(v).isnan().item() else v  # turn nan to str('nan_value')
            ) for k, v in one_metric.items()
        }
        self.history.append(one_metric)
        logging.info(
            pformat(
                one_metric
            )
        )

    def to_dataframe(self):
        self.df = pd.DataFrame(self.history, dtype=object)
        logging.info("return df with np.nan and None converted by str()")
        return self.df

    def summary(self):
        '''
        do summary for dataframe of record
        :return:
        eg.
            ,train_epoch_num,train_acc_clean
            last,100.0,96.68965148925781
            valid_max,100.0,96.70848846435547
            exist_nan_value,False,False
        '''
        if 'df' not in self.__dict__:
            logging.info('No df found in Metric_Aggregator, generate now')
            self.to_dataframe()
        logging.info("return df with np.nan and None converted by str()")
        return self.df.apply(last_and_valid_max)


class ModelTrainerCLS():
    def __init__(self, model, amp=False):
        self.model = model
        self.amp = amp

    def init_or_continue_train(self,
                               train_data,
                               end_epoch_num,
                               criterion,
                               optimizer,
                               scheduler,
                               device,
                               continue_training_path: Optional[str] = None,
                               only_load_model: bool = False,
                               ) -> None:
        '''
        config the training process, from 0 or continue previous.
        The requirement for saved file please refer to save_all_state_to_path
        :param train_data: train_data_loader, only if when you need of number of batch, you need to input it. Otherwise just skip.
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.

        '''

        model = self.model

        model.to(device)
        model.train()

        # train and update

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        if continue_training_path is not None:
            logging.info(f"No batch info will be used. Cannot continue from specific batch!")
            # start_epoch, start_batch = self.load_from_path(continue_training_path, device, only_load_model)
            # if (start_epoch is None) or (start_batch is None):
            #     self.start_epochs, self.end_epochs = 0, end_epoch_num
            #     self.start_batch = 0
            # else:
            #     batch_num = len(train_data)
            #     self.start_epochs, self.end_epochs = start_epoch + ((start_batch + 1)//batch_num), end_epoch_num
            #     self.start_batch = (start_batch + 1) % batch_num
            start_epoch, _ = self.load_from_path(continue_training_path, device, only_load_model)
            self.start_epochs, self.end_epochs = start_epoch, end_epoch_num
        else:
            self.start_epochs, self.end_epochs = 0, end_epoch_num
            # self.start_batch = 0

        logging.info(f'All setting done, train from epoch {self.start_epochs} to epoch {self.end_epochs}')

        logging.info(
            pformat(f"self.amp:{self.amp}," +
                    f"self.criterion:{self.criterion}," +
                    f"self.optimizer:{self.optimizer}," +
                    f"self.scheduler:{self.scheduler.state_dict() if self.scheduler is not None else None}," +
                    f"self.scaler:{self.scaler.state_dict() if self.scaler is not None else None})")
        )

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_all_state_to_path(self,
                               path: str,
                               epoch: Optional[int] = None,
                               batch: Optional[int] = None,
                               only_model_state_dict: bool = False) -> None:
        '''
        save all information needed to continue training, include 3 random state in random, numpy and torch
        :param path: where to save
        :param epoch: which epoch when save
        :param batch: which batch index when save
        :param only_model_state_dict: only save the model, drop all other information
        '''

        save_dict = {
            'epoch_num_when_save': epoch,
            'batch_num_when_save': batch,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.random.get_rng_state(),
            'model_state_dict': self.get_model_params(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'criterion_state_dict': self.criterion.state_dict(),
            "scaler": self.scaler.state_dict(),
        } \
            if only_model_state_dict == False else self.get_model_params()

        torch.save(
            save_dict,
            path,
        )

    def load_from_path(self,
                       path: str,
                       device,
                       only_load_model: bool = False
                       ) -> [Optional[int], Optional[int]]:
        '''

        :param path:
        :param device: map model to which device
        :param only_load_model: only_load_model or not?
        '''

        self.model = self.model.to(device)

        load_dict = torch.load(
            path, map_location=device
        )

        logging.info(f"loading... keys:{load_dict.keys()}, only_load_model:{only_load_model}")

        attr_list = [
            'epoch_num_when_save',
            'batch_num_when_save',
            'random_state',
            'np_random_state',
            'torch_random_state',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'criterion_state_dict',
        ]

        if all([key_name in load_dict for key_name in attr_list]):
            # all required key can find in load dict
            # AND only_load_model == False
            if only_load_model == False:
                random.setstate(load_dict['random_state'])
                np.random.set_state(load_dict['np_random_state'])
                torch.random.set_rng_state(load_dict['torch_random_state'].cpu())  # since may map to cuda

                self.model.load_state_dict(
                    load_dict['model_state_dict']
                )
                self.optimizer.load_state_dict(
                    load_dict['optimizer_state_dict']
                )
                if self.scheduler is not None:
                    self.scheduler.load_state_dict(
                        load_dict['scheduler_state_dict']
                    )
                self.criterion.load_state_dict(
                    load_dict['criterion_state_dict']
                )
                if 'scaler' in load_dict:
                    self.scaler.load_state_dict(
                        load_dict["scaler"]
                    )
                    logging.info(f'load scaler done. scaler={load_dict["scaler"]}')
                logging.info('all state load successful')
                return load_dict['epoch_num_when_save'], load_dict['batch_num_when_save']
            else:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None

        else:  # only state_dict

            if 'model_state_dict' in load_dict:
                self.model.load_state_dict(
                    load_dict['model_state_dict'],
                )
                logging.info('only model state_dict load')
                return None, None
            else:
                self.model.load_state_dict(
                    load_dict,
                )
                logging.info('only model state_dict load')
                return None, None

    def test(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            # 'detail_list' : [],
        }

        criterion = self.criterion.to(device)

        with torch.no_grad():
            for batch_idx, (x, target, *additional_info) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target.long())

                # logging.info(list(zip(additional_info[0].cpu().numpy(), pred.detach().cpu().numpy(),
                #                target.detach().cpu().numpy(), )))

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    # @resource_check
    def train_one_batch(self, x, labels, device):
        self.model.train()
        self.model.to(device)
        labels = labels.long()
        x, labels = x.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(x)
            loss = self.criterion(output, labels.long())
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        batch_loss = loss.item() * labels.size(0)

        return batch_loss, output

    def train_one_batch_dynamic_ls(self, x, labels, smooth_rates, device):
        self.model.train()
        self.model.to(device)
        labels = labels.long()
        x, labels = x.to(device), labels.to(device)
        smooth_rates = smooth_rates.to(device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            output = self.model(x)
            loss = self.criterion(output, labels.long(), smooth_rates)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        batch_loss = loss.item() * labels.size(0)
        return batch_loss, output

    def train_one_epoch(self, train_data, device, dynamic_ls=False, aum_accumulator=None, omega_alignment=None):
        startTime = time()
        batch_loss = []
        if dynamic_ls:
            for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
                ori_index, _, _, smooth_rates = additional_info
                loss, logits = self.train_one_batch_dynamic_ls(x, labels, smooth_rates, device)
                batch_loss.append(loss)
                if aum_accumulator is not None:
                    aum_accumulator.oneBatchAccumulateInOneEpoch(logits, labels, ori_index)
        else:
            for batch_idx, (x, labels, *additional_info) in enumerate(train_data):
                loss, logits = self.train_one_batch(x, labels, device)
                batch_loss.append(loss)
                if aum_accumulator is not None:
                    ori_index, _, _ = additional_info
                    aum_accumulator.oneBatchAccumulateInOneEpoch(logits, labels, ori_index)
                # if omega_alignment is not None:

        if aum_accumulator is not None:
            print(f'Getting AUM......')
            aum_accumulator.oneBatchAccumulateDown()

        one_epoch_loss = sum(batch_loss)
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # here since ReduceLROnPlateau need the train loss to decide next step setting.
                self.scheduler.step(one_epoch_loss)
            else:
                self.scheduler.step()

        endTime = time()

        logging.info(f"one epoch training part done, use time = {endTime - startTime} s")

        return one_epoch_loss

    def train(self, train_data, end_epoch_num,
              criterion,
              optimizer,
              scheduler, device, frequency_save, save_folder_path,
              save_prefix,
              continue_training_path: Optional[str] = None,
              only_load_model: bool = False, ):
        '''

        simplest train algorithm with init function put inside.

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''

        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train, epoch_loss: {epoch_loss[-1]}')
            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")

    def train_with_test_each_epoch(self,
                                   train_data,
                                   test_data,
                                   adv_test_data,
                                   end_epoch_num,
                                   criterion,
                                   optimizer,
                                   scheduler,
                                   device,
                                   frequency_save,
                                   save_folder_path,
                                   save_prefix,
                                   continue_training_path: Optional[str] = None,
                                   only_load_model: bool = False,
                                   ):
        '''
        train with test on benign and backdoor dataloader for each epoch

        :param train_data: train_data_loader
        :param test_data: benign test data
        :param adv_test_data: backdoor poisoned test data (for ASR)
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            one_epoch_loss = self.train_one_epoch(train_data, device)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            metrics = self.test(test_data, device)
            metric_info = {
                'epoch': epoch,
                'benign acc': metrics['test_correct'] / metrics['test_total'],
                'benign loss': metrics['test_loss'],
            }
            agg(metric_info)

            adv_metrics = self.test(adv_test_data, device)
            adv_metric_info = {
                'epoch': epoch,
                'ASR': adv_metrics['test_correct'] / adv_metrics['test_total'],
                'backdoor loss': adv_metrics['test_loss'],
            }
            agg(adv_metric_info)

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

    def train_with_test_each_epoch_v2(self,
                                      train_data,
                                      test_dataloader_dict,
                                      end_epoch_num,
                                      criterion,
                                      optimizer,
                                      scheduler,
                                      device,
                                      frequency_save,
                                      save_folder_path,
                                      save_prefix,
                                      continue_training_path: Optional[str] = None,
                                      only_load_model: bool = False,
                                      dynamic_ls: bool = False,
                                      p_rate: float = 0.2,
                                      n_rate: float = -1.0,
                                      warmup_end: int = 60,
                                      warmup_rate: float = 0.1,
                                      log_aum: bool = False,
                                      nls: bool = False,
                                      args=None,
                                      ):
        '''
        v2 can feed many test_dataloader, so easier for test with multiple dataloader.

        only change the test data part, instead of predetermined 2 dataloader, you can input any number of dataloader to test
        with {
            test_name (will show in log): test dataloader
        }
        in log you will see acc and loss for each test dataloader

        :param test_dataloader_dict: { name : dataloader }

        :param train_data: train_data_loader
        :param end_epoch_num: end training epoch number, if not continue training process, then equal to total training epoch
        :param criterion: loss function used
        :param optimizer: optimizer
        :param scheduler: scheduler
        :param device: device
        :param frequency_save: how many epoch to save model and random states information once
        :param save_folder_path: folder path to save files
        :param save_prefix: for saved files, the prefix of file name
        :param continue_training_path: where to load files for continue training process
        :param only_load_model: only load the model, do not load other settings and random state.
        '''
        if log_aum:
            aum_accumulator = AreaUnderTheMarginRanking(train_data, self.model, device)
            print('logging AUM')
        else:
            aum_accumulator = None

        agg = Metric_Aggregator()
        self.init_or_continue_train(
            train_data,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            continue_training_path,
            only_load_model
        )
        epoch_loss = []
        for epoch in range(self.start_epochs, self.end_epochs):
            if nls:
                if epoch == 0:
                    train_data.dataset.set_dynamic_smooth_rate(warmup_rate, warmup_rate)
                if epoch == warmup_end - 1:
                    train_data.dataset.set_dynamic_smooth_rate(p_rate, n_rate)
            if dynamic_ls:
                if epoch == 0:
                    train_data.dataset.set_dynamic_smooth_rate(warmup_rate, warmup_rate)
                if epoch == warmup_end - 1:
                    aum_accumulator.logging_aumAtCurrentEpoch()
                    logging.info('Set By Mean')
                    train_data.dataset.set_dynamic_smooth_rate_ls_aum(p_rate,
                                                                      n_rate,
                                                                      aum_accumulator.get_aumAtCurrentEpoch())
                    import platform
                    import os
                    plat = platform.system().lower()
                    if plat == 'windows':
                        absolute_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    # linux
                    else:
                        absolute_path = os.getcwd()
                        print(absolute_path)
                        absolute_path = os.path.dirname(absolute_path)
                        print(absolute_path)
                    save_dir = os.path.join(absolute_path, args.save_path[3:], 'aum_density')
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    adv, benign = aum_accumulator.get_aumAtCurrentEpochOfTargetClass(args.attack_target)
                    np.save(os.path.join(save_dir, f'label{args.attack_target}-adv_aum.npy'), adv)
                    np.save(os.path.join(save_dir, f'label{args.attack_target}-benign_aum.npy'), benign)
                    legend_list = ['Benign Samples', 'Poisoned Samples']
                    density_plot([np.array(benign), np.array(adv)], legend_list,
                                 os.path.join(save_dir, f'epoch_{epoch}-label{args.attack_target}-aum_density.pdf'))

            one_epoch_loss = self.train_one_epoch(train_data, device, dynamic_ls,
                                                  aum_accumulator)
            epoch_loss.append(one_epoch_loss)
            logging.info(f'train_with_test_each_epoch, epoch:{epoch} ,epoch_loss: {epoch_loss[-1]}')

            if aum_accumulator is not None:
                aum_accumulator.logging_aumAtCurrentEpochOfTargetClass(args.attack_target)
                # aum_accumulator.logging_SelfEntropyAtCurrentEpochOfTargetClass(args.attack_target)

            for dl_name, test_dataloader in test_dataloader_dict.items():
                metrics = self.test(test_dataloader, device)
                metric_info = {
                    'epoch': epoch,
                    f'{dl_name} acc': metrics['test_correct'] / metrics['test_total'],
                    f'{dl_name} loss': metrics['test_loss'],
                }
                agg(metric_info)

            # if log_aum:
            #     print(f'Getting AUM......')
            #     aum_accumulator.epochAccumulate()
            #     print(f'Got AUM at Epoch {epoch}')

            if frequency_save != 0 and epoch % frequency_save == frequency_save - 1:
                logging.info(f'saved. epoch:{epoch}')
                self.save_all_state_to_path(
                    epoch=epoch,
                    path=f"{save_folder_path}/{save_prefix}_epoch_{epoch}.pt")
            # logging.info(f"training, epoch:{epoch}, batch:{batch_idx},batch_loss:{loss.item()}")
            agg.to_dataframe().to_csv(f"{save_folder_path}/{save_prefix}_df.csv")
        agg.summary().to_csv(f"{save_folder_path}/{save_prefix}_df_summary.csv")

        if log_aum:
            print(f'Logging AUM:')
            aum_accumulator.logging_aumAtCurrentEpoch()

    def train_with_test_each_epoch_v2_sp(self,
                                         batch_size,
                                         train_dataset,
                                         test_dataset_dict,
                                         end_epoch_num,
                                         criterion,
                                         optimizer,
                                         scheduler,
                                         device,
                                         frequency_save,
                                         save_folder_path,
                                         save_prefix,
                                         continue_training_path: Optional[str] = None,
                                         only_load_model: bool = False,
                                         dynamic_ls: bool = False,
                                         p_rate=0.2,
                                         n_rate=0,
                                         warmup_end=10,
                                         warmup_rate=0.1,
                                         args=None,
                                         log_aum: bool = False,
                                         nls: bool = False
                                         ):

        '''
        Nothing different, just be simplified to accept dataset instead.
        '''

        train_data = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_dataloader_dict = {
            name: DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )
            for name, test_dataset in test_dataset_dict.items()
        }

        self.train_with_test_each_epoch_v2(
            train_data,
            test_dataloader_dict,
            end_epoch_num,
            criterion,
            optimizer,
            scheduler,
            device,
            frequency_save,
            save_folder_path,
            save_prefix,
            continue_training_path,
            only_load_model,
            dynamic_ls,
            p_rate,
            n_rate,
            warmup_end,
            warmup_rate,
            log_aum,
            nls,
            args=args,
        )
