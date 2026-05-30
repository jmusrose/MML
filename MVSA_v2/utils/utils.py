#!/usr/bin/env python3
"""
Utility functions: seed, checkpoint, metrics, averager.
"""

import json
import os
import random
import shutil

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    if torch is None:
        raise RuntimeError("torch is required for set_seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_path, filename="checkpoint.pt"):
    """Save model checkpoint and optionally copy as best model."""
    if torch is None:
        raise RuntimeError("torch is required for save_checkpoint")
    filename = os.path.join(checkpoint_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_path, "model_best.pt"))


def load_checkpoint(model, path):
    """Load model state dict from checkpoint file."""
    if torch is None:
        raise RuntimeError("torch is required for load_checkpoint")
    best_checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(best_checkpoint["state_dict"])


class Averager(object):
    """Running average tracker for loss values."""

    def __init__(self):
        self.n = 0
        self.v = 0.0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

    def reset(self):
        self.n = 0
        self.v = 0.0


def store_preds_to_disk(tgts, preds, args):
    """Write predictions and gold labels to disk."""
    with open(os.path.join(args.savedir, "test_labels_pred.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in preds]))
    with open(os.path.join(args.savedir, "test_labels_gold.txt"), "w") as fw:
        fw.write("\n".join([str(x) for x in tgts]))
    with open(os.path.join(args.savedir, "test_labels.txt"), "w") as fw:
        fw.write(" ".join([str(l) for l in args.labels]))


def log_metrics(set_name, metrics, args, logger):
    """Log evaluation metrics."""
    logger.info(
        "{}: Loss: {:.5f} | Acc: {:.5f}".format(
            set_name, metrics["loss"], metrics["acc"]
        )
    )


def append_experiment_record(summary_path: str, record: dict) -> None:
    """Append one experiment record to a JSON-array summary file."""
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
