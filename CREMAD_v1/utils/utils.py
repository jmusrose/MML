#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torchvision import transforms

import os
from os import path as osp

import logging
import numpy as np
from datetime import datetime


def deep_update_dict(fr, to):
    '''update dict of dicts with new values'''
    for k, v in fr.items():
        if isinstance(v, dict):
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def create_logger(cfg, rank=0, test=False):
    dataset = cfg['dataset']['dataset_name']
    backbone_name = cfg['visual']['name'] + ' ' + cfg['text']['name']
    head_type = cfg.get('head', {}).get('type', 'MLP')

    if test:
        log_dir = osp.join(cfg.get('output_dir', '.'), dataset, "test")
        log_name = '{}.log'.format(cfg['test'].get('exp_id', 'test'))
        log_file = osp.join(log_dir, log_name)
    else:
        log_dir = osp.join(cfg.get('output_dir', '.'), dataset, "logs")
        time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        loss = cfg['loss']['type']
        seed = cfg['seed']
        log_name = "{}_{}_{}_{}_{}_{}.log".format(dataset, backbone_name, loss, seed, head_type, time_str)
        log_file = osp.join(log_dir, log_name)

    if not osp.exists(log_dir) and rank == 0:
        os.makedirs(log_dir)

    print("=> creating log {}".format(log_file))
    header = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=header)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file, log_name.split('.')[0]


def get_scheduler(cfg, optimizer, t_max=None):
    scheduler_type = cfg['train']['lr_scheduler'].get('type', 'normal')
    if scheduler_type == 'normal':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg['train']['lr_scheduler']['patience'], 0.1
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=t_max if t_max else cfg['train']['epoch_dict'],
            eta_min=0,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(scheduler_type))
    return scheduler


def param_count(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k
