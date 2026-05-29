#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import torch
except ModuleNotFoundError:
    torch = None

import json
import os
import shutil
import time
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

    if cfg.get('run_dir') and not test:
        log_dir = cfg['run_dir']
        log_name = cfg.get('log_name', 'training.log')
        log_file = osp.join(log_dir, log_name)
    elif test:
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
    if torch is None:
        raise RuntimeError("torch is required for get_scheduler")
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


def append_experiment_record(summary_path: str, record: dict) -> None:
    """Append one experiment record to a JSON-array summary file."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    records = []
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                records = loaded
            else:
                raise ValueError(f"Existing summary is not a list: {type(loaded).__name__}")
        except Exception as exc:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{summary_path}.corrupt-{ts}.bak"
            shutil.copyfile(summary_path, backup)
            print(
                f"[append_experiment_record] {summary_path} unreadable ({exc}); "
                f"backed up to {backup}, reinitializing."
            )

    records.append(record)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
