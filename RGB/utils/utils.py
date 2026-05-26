#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Common utilities for the DML_v1/RGB project.
#
# Style follows CPSC_RGB/utils/utils.py and CPSC_RGB/CPSC_nyu.py:
# - set_seed sets random / numpy / torch / cuda seeds and forces deterministic
#   cuDNN behaviour (deterministic=True, benchmark=False) so two runs with the
#   same seed yield identical results.
# - save_checkpoint persists training state via torch.save and additionally
#   copies the file to model_best.pt when is_best is True.
# - load_checkpoint loads a checkpoint produced by save_checkpoint and restores
#   it onto the given model. It accepts both the {'model_state_dict': ...}
#   layout used by this project (see Requirement 10.1) and the legacy
#   {'state_dict': ...} layout used by CPSC_RGB.
# - Averager accumulates a running mean; item() returns 0.0 before any add.

import os
import random
import shutil

import numpy as np
import torch


class Averager:
    """Running-mean accumulator.

    Usage:
        avg = Averager()
        avg.add(0.5)
        avg.add(1.5)
        avg.item()   # -> 1.0
        avg.reset()
        avg.item()   # -> 0.0
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value):
        self.sum += float(value)
        self.count += 1

    def item(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def set_seed(seed):
    """Seed every PRNG we touch and force deterministic cuDNN.

    Covers Python ``random``, NumPy, CPU torch, current-device CUDA torch and
    all CUDA devices. Also disables cuDNN auto-tuner so kernel choice does not
    drift between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    """Persist ``state`` to ``checkpoint_path/filename`` and mirror to model_best.pt when best."""
    os.makedirs(checkpoint_path, exist_ok=True)
    full_path = os.path.join(checkpoint_path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    """Load a checkpoint from ``path`` and restore weights onto ``model``.

    Supports the project checkpoint layout
    ``{'epoch', 'model_state_dict', 'optimizer_state_dict', 'clean_acc'}`` as
    well as the legacy ``{'state_dict': ...}`` layout. If neither key is
    present, the loaded object itself is treated as the state dict.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    return checkpoint


def append_experiment_record(summary_path: str, record: dict) -> None:
    """将一条实验记录追加到 ``summary_path`` 指向的 JSON 数组文件。

    行为
    ----
    - 若文件不存在或目录尚未创建，先 ``os.makedirs`` 并以 ``[record]`` 初始化。
    - 若文件存在但解析失败 / 不是 list，会备份原文件为 ``<name>.corrupt-<ts>.bak``
      后用 ``[record]`` 重新初始化，避免一次解析失败丢掉所有记录。
    - 写入使用 ``indent=4`` + ``ensure_ascii=False`` 以保留中文备注可读性。

    设计上属于"尽力而为"的简单实现：不做文件锁，多进程并发写入可能丢记录，
    研究脚本场景下可以接受。
    """
    import json
    import time

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
        except Exception as e:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = f"{summary_path}.corrupt-{ts}.bak"
            shutil.copyfile(summary_path, backup)
            print(f"[append_experiment_record] {summary_path} unreadable ({e}); backed up to {backup}, reinitializing.")
            records = []

    records.append(record)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
